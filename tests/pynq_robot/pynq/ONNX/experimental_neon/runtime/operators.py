"""Host tensor API with reusable CMA staging; actual GEMM/Softmax execution on PL."""
import sys
import gc
from collections import OrderedDict
from pathlib import Path
import time
import threading
import os
import numpy as np
from pynq import allocate

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from fast_gemm import FastGemm
from accelerator_driver import AcceleratorDriver
from .neon_runtime import FastCpu,FastPack


def up32(n):return (int(n)+31)//32*32

DEFAULT_CPU_THRESHOLD_OPS=12058624


class PynqOperators:
    def __init__(self,hwh=None,cpu_threshold_ops=DEFAULT_CPU_THRESHOLD_OPS,
                 resident_cache_slots=None,resident_cache_bytes=64*1024**2,
                 softmax_buffer_elements=1024**2,shared_scratch=False):
        self.buffers=[]
        self.fast=None;self.control=None
        self.shared_scratch=bool(shared_scratch)
        self._scratch_lock=threading.RLock() if self.shared_scratch else None
        self._scratch_owner=None
        self._softmax_overwrites_bt=False
        self.resident_cache_slots=resident_cache_slots or []
        self.resident_cache_bytes=int(resident_cache_bytes) if self.resident_cache_slots else 0
        self.resident_cache_loaded=set()
        self.resident_cache_by_id={int(row['cache_id']):row for row in self.resident_cache_slots}
        if len(self.resident_cache_by_id)!=len(self.resident_cache_slots):
            raise ValueError('Resident CMA cache IDs must be unique')
        if self.resident_cache_bytes<0:raise ValueError('Resident cache size must be nonnegative')
        self.metrics={}
        self.plans={}
        self.a_pack_mode=os.environ.get('SMOL_A_PACK_MODE','full')
        if self.a_pack_mode not in ('full','edges','neon','adaptive'):
            raise ValueError('SMOL_A_PACK_MODE must be full, edges, neon, or adaptive')
        self.a_packer=None
        self.cpu_threshold_ops=int(cpu_threshold_ops)
        if self.cpu_threshold_ops<0:raise ValueError('cpu_threshold_ops must be nonnegative')
        self.softmax_buffer_elements=max(1024**2,int(softmax_buffer_elements))
        self.cpu_plans={}
        self.bt_cache=OrderedDict()
        self.bt_cache_owners=[None,None,None]
        self.hwh=Path(hwh) if hwh else ROOT/'hardware/design_1.hwh'
        try:
            mib=1024**2
            sm_bytes=self.softmax_buffer_elements*np.dtype(np.float32).itemsize
            separate_bytes=28*mib+2*sm_bytes
            if self.shared_scratch:
                # Only owners belong to buffers; views must never be freed
                # separately. GEMM and Softmax are mutually exclusive below.
                self.scratch_bytes=max(28*mib,2*sm_bytes)
                arena=allocate((self.scratch_bytes,),dtype=np.uint8,cacheable=1)
                self.buffers.append(arena)
                self.a=arena[:4*mib].view(np.int8)
                self.bt=arena[4*mib:20*mib].view(np.int8)
                self.c=arena[20*mib:28*mib].view(np.int32)
                self.sin=arena[:sm_bytes].view(np.float32)
                self.sout=arena[self.scratch_bytes-sm_bytes:].view(np.float32)
                # Prefer A/C storage. Larger Softmax pools also overlap BT;
                # invalidate its transient cache before those bytes change.
                self._softmax_overwrites_bt=(sm_bytes>4*mib or
                    self.scratch_bytes-sm_bytes<20*mib)
            else:
                self.scratch_bytes=separate_bytes
                for count,dtype in [(4*mib,np.int8),(16*mib,np.int8),
                                    (2*mib,np.int32),
                                    (self.softmax_buffer_elements,np.float32),
                                    (self.softmax_buffer_elements,np.float32)]:
                    self.buffers.append(allocate((count,),dtype=dtype,cacheable=1))
                self.a,self.bt,self.c,self.sin,self.sout=self.buffers[:5]
            self.scratch_bytes_saved=separate_bytes-self.scratch_bytes
            if self.resident_cache_bytes:
                required=max((int(row['offset_bytes'])+int(row['size_bytes'])
                              for row in self.resident_cache_slots),default=0)
                if required>self.resident_cache_bytes:
                    raise ValueError('Resident weight slots exceed configured CMA cache bytes')
                self.resident_cache_buffer=allocate((self.resident_cache_bytes,),dtype=np.int8,cacheable=1)
                self.buffers.append(self.resident_cache_buffer)
            else:self.resident_cache_buffer=None
            # This control object submits Softmax only. FastGemm owns all GEMMs.
            self.control=AcceleratorDriver.from_hwh(self.hwh,poll_interval=0)
            self.fast=FastGemm(self.hwh,backend='native')
            self.reset_metrics()
        except BaseException:
            self.close();raise

    def reset_metrics(self):
        self.metrics=dict(gemm_calls=0,gemm_tasks=0,gemm_groups=0,gemm_core_jobs=0,
                          cpu_gemm_calls=0,cpu_gemm_seconds=0.0,
                          weight_transpose_cache_hits=0,weight_transpose_cache_misses=0,
                          resident_weight_cache_hits=0,resident_weight_cache_fills=0,
                          resident_weight_cache_bytes_copied=0,resident_weight_cache_syncs=0,
                          softmax_calls=0,softmax_tasks=0,gemm_seconds=0.0,
                          softmax_seconds=0.0,logical_gemm_ops=0,
                          shared_scratch_switches=0,shared_scratch_sync_bytes=0,
                          shared_scratch_sync_seconds=0.0,
                          a_pack_seconds=0.0,a_padded_bytes_written=0,
                          a_padding_bytes_zeroed=0)

    def _pack_a(self,source,destination,rows,k,padded_rows,padded_k):
        mode=self.a_pack_mode
        if mode=='adaptive':
            padded_bytes=padded_rows*padded_k
            mode=('full' if padded_bytes < 64*1024 else
                  'edges' if padded_bytes < 200*1024 else 'neon')
        if mode=='full':
            destination.fill(0)
            np.subtract(source,np.uint8(128),out=destination[:rows,:k].view(np.uint8))
        elif mode=='edges':
            np.subtract(source,np.uint8(128),out=destination[:rows,:k].view(np.uint8))
            if padded_k>k:destination[:rows,k:padded_k].fill(0)
            if padded_rows>rows:destination[rows:padded_rows,:padded_k].fill(0)
        else:
            if self.a_packer is None:
                from .pack_a_neon import NeonAPacker
                self.a_packer=NeonAPacker()
            self.a_packer(source,destination,rows,k,padded_rows,padded_k)

    def _enter_shared_scratch(self,kind):
        # A timeout retains live hardware work. Even after its method exits,
        # the other IP must not overwrite that task's borrowed arena storage.
        if self.fast.active is not None or self.control.pending:
            raise RuntimeError('Shared CMA scratch has pending hardware work; drain it before reuse')
        if self._scratch_owner!=kind:
            if self._scratch_owner is None:
                # Clean possible allocator dirt once. Thereafter all CPU writes
                # are A/BT/sin inputs, explicitly flushed before START; C/sout
                # are only read by CPU after invalidation. Completed operations
                # therefore leave no dirty lines to overwrite the next IP's
                # output, including when its view reuses another input span.
                start=time.perf_counter()
                self.buffers[0].sync_to_device()
                self.metrics['shared_scratch_sync_bytes']+=self.scratch_bytes
                self.metrics['shared_scratch_sync_seconds']+=time.perf_counter()-start
            self.metrics['shared_scratch_switches']+=1
            self._scratch_owner=kind

    def _activate_softmax_scratch(self):
        """Prepare direct Softmax buffer use; caller must hold _scratch_lock if set."""
        if self.shared_scratch:
            self._enter_shared_scratch('softmax')
        if self._softmax_overwrites_bt:
            self.bt_cache.clear();self.bt_cache_owners=[None,None,None]

    @staticmethod
    def _bt_key(b,pretransposed=False):
        return (int(b.ctypes.data),tuple(b.shape),tuple(b.strides),b.dtype.str,bool(pretransposed))

    def _prepare_bt(self,b,core,constant_b,slot=None,allow_full=False,pretransposed=False):
        if pretransposed:
            if b.ndim!=2:raise ValueError('Pretransposed constant B must be a 2-D [N,K] matrix')
            n,k=b.shape
        else:n,k=b.T.shape
        kp,npad=up32(k),up32(n)
        count=kp*npad
        slot_bytes=self.bt.nbytes//3//64*64
        use_full=allow_full and count>slot_bytes
        key=self._bt_key(b,pretransposed) if constant_b else None
        owner=self.bt_cache.get(key) if key is not None else None
        if owner is not None:
            if (owner==-1 and use_full) or (owner>=0 and not use_full):
                self.bt_cache.move_to_end(key)
                self.metrics['weight_transpose_cache_hits']+=1
                target=self.bt[:count] if owner==-1 else self._slot(self.bt,owner,np.int8)[:count]
                return target.reshape(npad,kp),True
            self.bt_cache.pop(key,None)
        if use_full:
            # Large single-job weights may use the whole pool; all slot caches overlap it.
            self.bt_cache.clear();self.bt_cache_owners=[None,None,None]
            owner=-1;bt=self.bt[:count].reshape(npad,kp)
            if constant_b:
                self.bt_cache[key]=owner
                self.metrics['weight_transpose_cache_misses']+=1
        else:
            # A slot write overlaps and invalidates any prior whole-pool transpose.
            for stale in [k for k,v in self.bt_cache.items() if v==-1]:self.bt_cache.pop(stale,None)
            owner=core if slot is None else slot
            previous=self.bt_cache_owners[owner]
            if previous is not None:self.bt_cache.pop(previous,None)
            if constant_b:
                self.bt_cache[key]=owner;self.bt_cache_owners[owner]=key
                self.metrics['weight_transpose_cache_misses']+=1
            else:self.bt_cache_owners[owner]=None
            bt=self._slot(self.bt,owner,np.int8)[:count].reshape(npad,kp)
        bt.fill(0)
        if pretransposed:
            np.copyto(bt[:n,:k],b)
        elif k%8==0 and n%8==0:
            FastPack(np.ascontiguousarray(b),np.asarray(bt)).run(None,dict(mode=1,block=64,threads=4))
        else:np.copyto(bt[:n,:k],b.T)
        return bt,False

    def _resident_bt(self,b,cache_id,n,k):
        cache_id=int(cache_id)
        if cache_id<0 or self.resident_cache_buffer is None:
            raise ValueError('Invalid resident cache ID or cache not initialized')
        row=self.resident_cache_by_id.get(cache_id)
        if row is None:raise ValueError('Unknown resident cache ID {}'.format(cache_id))
        padded_n,padded_k=up32(n),up32(k)
        if tuple(row.get('padded_shape',()))!=(padded_n,padded_k):
            raise ValueError('Resident cache shape mismatch for ID {}'.format(cache_id))
        size=int(row['size_bytes']);offset=int(row['offset_bytes'])
        if size!=padded_n*padded_k or offset<0 or offset+size>self.resident_cache_buffer.size:
            raise ValueError('Invalid resident CMA cache slot {}'.format(cache_id))
        target=self.resident_cache_buffer[offset:offset+size].reshape(padded_n,padded_k)
        if cache_id in self.resident_cache_loaded:
            self.metrics['resident_weight_cache_hits']+=1
            return target,True
        if b.shape!=(n,k):raise ValueError('Resident B must be pretransposed [N,K]')
        target.fill(0);np.copyto(target[:n,:k],b)
        target.sync_to_device()
        self.resident_cache_loaded.add(cache_id)
        self.metrics['resident_weight_cache_fills']+=1
        self.metrics['resident_weight_cache_bytes_copied']+=int(b.nbytes)
        self.metrics['resident_weight_cache_syncs']+=1
        return target,True

    def _gemm_matrix(self,a,b,out,constant_b=False,cache_id=-1):
        m,k=a.shape
        pretransposed=constant_b==2
        n,bk=b.shape if pretransposed else (b.shape[1],b.shape[0])
        if k!=bk:raise ValueError('K mismatch')
        kp,npad=up32(k),up32(n)
        if kp>16352 or kp*npad>self.bt.size:
            raise ValueError('GEMM shape exceeds staging pool')
        if cache_id>=0 and int(cache_id) in self.resident_cache_by_id:
            if not pretransposed:raise ValueError('Resident cache requires pretransposed static B')
            bt,bt_cached=self._resident_bt(b,cache_id,n,k)
        else:
            bt,bt_cached=self._prepare_bt(b,0,constant_b,allow_full=True,pretransposed=pretransposed)
        if not bt_cached:bt.sync_to_device()
        rows=min((self.a.size//kp)//32*32,(self.c.size//npad)//32*32,65535*32)
        if rows<32:raise ValueError('GEMM row tile does not fit')
        for first in range(0,m,rows):
            nr=min(rows,m-first);mp=up32(nr)
            aa=self.a[:mp*kp].reshape(mp,kp)
            cc=self.c[:mp*npad].reshape(mp,npad)
            pack_start=time.perf_counter()
            self._pack_a(a[first:first+nr],aa,nr,k,mp,kp)
            self.metrics['a_pack_seconds']+=time.perf_counter()-pack_start
            self.metrics['a_padded_bytes_written']+=mp*kp
            self.metrics['a_padding_bytes_zeroed']+=mp*kp-nr*k
            aa.sync_to_device()
            key=(mp,npad,kp,int(aa.device_address),int(bt.device_address),int(cc.device_address))
            plan=self.plans.get(key)
            if plan is None:
                plan=self.fast.prepare(aa,bt,cc,(0,1,2));self.plans[key]=plan
            plan.submit().wait(timeout_s=30)
            cc.sync_from_device()
            np.copyto(out[first:first+nr],cc[:nr,:n])
            self.metrics['gemm_tasks']+=1
            self.metrics['gemm_groups']+=1
            self.metrics['gemm_core_jobs']+=len(plan.cores)
        self.metrics['logical_gemm_ops']+=2*m*n*k

    @staticmethod
    def _slot(buffer,core,dtype):
        dtype=np.dtype(dtype)
        slot_bytes=buffer.nbytes//3//64*64
        first=core*(slot_bytes//dtype.itemsize)
        return buffer.view(dtype)[first:first+slot_bytes//dtype.itemsize]

    def _gemm_group(self,matrices,constant_b=False):
        """Run up to three independent batch matrices in one hardware group."""
        prepared=[];batches=[];shared_bt=None;bt_synced=set()
        share_b=(len(matrices)>1 and
                 all(self._bt_key(item[1])==self._bt_key(matrices[0][1])
                     for item in matrices))
        assignments={};reserved=set()
        for core,(a,b,out) in enumerate(matrices):
            identity=self._bt_key(b)
            if identity in assignments:continue
            key=identity if constant_b else None
            owner=self.bt_cache.get(key) if key is not None else None
            if owner is not None:
                assignments[identity]=owner;reserved.add(owner)
        for core,(a,b,out) in enumerate(matrices):
            identity=self._bt_key(b)
            if identity in assignments:continue
            owner=core if core not in reserved else next(i for i in range(3) if i not in reserved)
            assignments[identity]=owner;reserved.add(owner)
        for core,(a,b,out) in enumerate(matrices):
            m,k=a.shape;bk,n=b.shape
            if k!=bk:raise ValueError('K mismatch')
            mp,kp,npad=up32(m),up32(k),up32(n)
            aa_pool=self._slot(self.a,core,np.int8)
            cc_pool=self._slot(self.c,core,np.int32)
            bt_pool=self._slot(self.bt,core,np.int8)
            if mp*kp>aa_pool.size or kp*npad>bt_pool.size or mp*npad>cc_pool.size:
                return False
            prepared.append((core,a,b,out,mp,kp,npad,aa_pool,bt_pool,cc_pool))

        jobs=[]
        for core,a,b,out,mp,kp,npad,aa_pool,bt_pool,cc_pool in prepared:
            m,k=a.shape;n=b.shape[1]
            aa=aa_pool[:mp*kp].reshape(mp,kp)
            pack_start=time.perf_counter()
            self._pack_a(a,aa,m,k,mp,kp)
            self.metrics['a_pack_seconds']+=time.perf_counter()-pack_start
            self.metrics['a_padded_bytes_written']+=mp*kp
            self.metrics['a_padding_bytes_zeroed']+=mp*kp-m*k
            if share_b:
                if shared_bt is None:
                    bt,bt_cached=self._prepare_bt(b,core,constant_b,assignments[self._bt_key(b)]);shared_bt=bt
                else:bt=shared_bt;bt_cached=True
            else:
                bt,bt_cached=self._prepare_bt(b,core,constant_b,assignments[self._bt_key(b)])
            cc=cc_pool[:mp*npad].reshape(mp,npad)
            aa.sync_to_device()
            address=int(bt.device_address)
            if not bt_cached and address not in bt_synced:
                bt.sync_to_device();bt_synced.add(address)
            jobs.append((core,aa,bt,cc))
            batches.append((m,n,out,cc))

        plan=self.fast.prepare_jobs(jobs)
        plan.submit().wait(timeout_s=30)
        for m,n,out,cc in batches:
            cc.sync_from_device();np.copyto(out,cc[:m,:n])
        self.metrics['gemm_tasks']+=1
        self.metrics['gemm_groups']+=1
        self.metrics['gemm_core_jobs']+=len(jobs)
        self.metrics['logical_gemm_ops']+=sum(2*a.shape[0]*b.shape[1]*a.shape[1] for a,b,_ in matrices)
        return True

    def _cpu_gemm(self,a,b,out):
        """Four-thread NEON fallback; avoid CMA copies and PL task latency."""
        start=time.perf_counter()
        m,k=a.shape;n=b.shape[1]
        mp,kp,npad=up32(m),up32(k),up32(n)
        key=(mp,npad,kp)
        state=self.cpu_plans.get(key)
        if state is None:
            aa=np.zeros((mp,kp),np.int8)
            bb=np.zeros((kp,npad),np.int8)
            cc=np.empty((mp,npad),np.int32)
            cpu=FastCpu(aa,bb,cc,workspace_pads=(0,))
            cpu.plan=dict(variant=2,mb=32,nb=128,pad=0,mode=1,block=32,pack_threads=4)
            state=(aa,bb,cc,cpu);self.cpu_plans[key]=state
        aa,bb,cc,cpu=state
        aa.fill(0);np.subtract(a,np.uint8(128),out=aa[:m,:k].view(np.uint8))
        bb.fill(0);np.copyto(bb[:k,:n],b)
        cpu.run(None)
        np.copyto(out,cc[:m,:n])
        self.metrics['cpu_gemm_calls']+=1
        self.metrics['cpu_gemm_seconds']+=time.perf_counter()-start
        self.metrics['logical_gemm_ops']+=2*a.shape[-2]*b.shape[-1]*a.shape[-1]

    def _should_use_cpu(self,a,b,constant_b=False):
        n,k=b.shape[-2:] if constant_b==2 else (b.shape[-1],b.shape[-2])
        ops=2*up32(a.shape[-2])*up32(n)*up32(a.shape[-1])
        return ops<=self.cpu_threshold_ops

    def _run_gemm_group(self,matrices,constant_b=False):
        cpu_jobs=[];pl_jobs=[]
        for a,b,out in matrices:
            (cpu_jobs if self._should_use_cpu(a,b,constant_b) else pl_jobs).append((a,b,out))
        for a,b,out in cpu_jobs:self._cpu_gemm(a,b,out)
        if len(pl_jobs)>1 and self._gemm_group(pl_jobs,constant_b):return
        for a,b,out in pl_jobs:self._gemm_matrix(a,b,out,constant_b)

    def matmul_integer(self,a,b,out,constant_b=False,cache_id=-1):
        if self._scratch_lock is not None:
            with self._scratch_lock:
                self._enter_shared_scratch('gemm')
                return self._matmul_integer(a,b,out,constant_b,cache_id)
        return self._matmul_integer(a,b,out,constant_b,cache_id)

    def _matmul_integer(self,a,b,out,constant_b=False,cache_id=-1):
        start=time.perf_counter()
        if a.dtype!=np.uint8 or b.dtype!=np.int8 or out.dtype!=np.int32:
            raise ValueError('Expected UINT8/INT8 -> INT32')
        if b.ndim==2:
            av=a.reshape(-1,a.shape[-1])
            ov=out.reshape(-1,out.shape[-1])
            if self._should_use_cpu(av,b,constant_b):
                self._cpu_gemm(av,b.T if constant_b==2 else b,ov)
            elif len(av)==0:
                raise ValueError('Empty MatMul rows')
            else:
                self._gemm_matrix(av,b,ov,constant_b,cache_id)
        else:
            batch=np.broadcast_shapes(a.shape[:-2],b.shape[:-2]) if hasattr(np,'broadcast_shapes') else np.broadcast(np.empty(a.shape[:-2]),np.empty(b.shape[:-2])).shape
            av=np.broadcast_to(a,batch+a.shape[-2:]);bv=np.broadcast_to(b,batch+b.shape[-2:])
            indices=list(np.ndindex(batch));ov=out.reshape((-1,out.shape[-2],out.shape[-1]))
            for first in range(0,len(indices),3):
                group=[(av[index],bv[index],ov[first+offset])
                       for offset,index in enumerate(indices[first:first+3])]
                self._run_gemm_group(group,constant_b)
        self.metrics['gemm_calls']+=1
        self.metrics['gemm_seconds']+=time.perf_counter()-start

    def softmax(self,x,out):
        if self._scratch_lock is not None:
            with self._scratch_lock:
                return self._softmax(x,out)
        return self._softmax(x,out)

    def _softmax(self,x,out):
        start=time.perf_counter()
        k=x.shape[-1];stride=(k+3)//4*4
        if not 1<=k<=4096:raise ValueError('Softmax K outside hardware limit')
        xx=x.reshape(-1,k);oo=out.reshape(-1,k)
        self._activate_softmax_scratch()
        rows=min(65536,self.sin.size//stride,self.sout.size//stride)
        for first in range(0,len(xx),rows):
            nr=min(rows,len(xx)-first)
            si=self.sin[:nr*stride].reshape(nr,stride)
            so=self.sout[:nr*stride].reshape(nr,stride)
            si.fill(0);np.copyto(si[:,:k],xx[first:first+nr])
            # The Softmax IP overwrites every element of the destination
            # span.  Flushing the old destination contents to DDR is both
            # unnecessary and costly; only flush the input before START and
            # invalidate the output after completion.
            si.sync_to_device()
            self.control.run_softmax(int(si.device_address),int(so.device_address),nr,k,timeout_s=30)
            so.sync_from_device();np.copyto(oo[first:first+nr],so[:,:k])
            self.metrics['softmax_tasks']+=1
        self.metrics['softmax_calls']+=1
        self.metrics['softmax_seconds']+=time.perf_counter()-start

    def close(self):
        if self.fast:
            while self.fast.active:
                try:self.fast.recover(timeout_s=30)
                except TimeoutError:time.sleep(.01)
            self.fast.close()
        if self.control:
            while self.control.pending:
                try:self.control.wait_all(timeout_s=30)
                except TimeoutError:time.sleep(.01)
            self.control.close()
        # Legacy Xlnk buffers release their ffi allocation only after every
        # NumPy view disappears; freebuffer() alone can just zero a pointer.
        # Retained routers may keep this closed operators object alive, so it
        # must not retain either cached plans or its public staging views.
        self.plans.clear();self.cpu_plans.clear();self.bt_cache.clear()
        self.bt_cache_owners=[None,None,None]
        self.resident_cache_loaded.clear()
        for b in reversed(self.buffers):b.freebuffer()
        self.buffers=[]
        self.a=self.bt=self.c=self.sin=self.sout=None
        self.resident_cache_buffer=None
        self.fast=self.control=None
        b=None
        gc.collect()
