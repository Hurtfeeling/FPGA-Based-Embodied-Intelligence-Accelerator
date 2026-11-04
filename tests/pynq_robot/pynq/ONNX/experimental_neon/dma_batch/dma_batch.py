"""Registered ordinary-memory <-> XLNK CMA copy, with explicit lifetime ownership."""
import fcntl
import os
import struct
import time
import numpy as np


def aligned_empty(shape,dtype):
    dtype=np.dtype(dtype)
    count=int(np.prod(shape))
    owner=np.empty(count*dtype.itemsize+63,np.uint8)
    offset=(-int(owner.ctypes.data))%64
    return owner[offset:offset+count*dtype.itemsize].view(dtype).reshape(shape)


class DmaCopy:
    def __init__(self,host,cma,to_cma,channel):
        if not isinstance(host,np.ndarray) or not isinstance(cma,np.ndarray):
            raise ValueError('NumPy host and PYNQ CMA arrays required')
        if (not host.flags.c_contiguous or not cma.flags.c_contiguous or
            host.nbytes!=cma.nbytes or host.ctypes.data%64 or cma.ctypes.data%64 or
            host.nbytes%64 or not hasattr(cma,'device_address') or
            getattr(cma,'freed',False)):
            raise ValueError('Live contiguous equally-sized buffers aligned to 64 bytes required')
        self.host,self.cma=host,cma
        self.to_cma=bool(to_cma)
        self.fd=os.open('/dev/ps_dma_batch',os.O_RDWR)
        start=time.perf_counter()
        try:
            fcntl.ioctl(self.fd,0xD201,struct.pack('=QII',host.nbytes,int(to_cma),channel))
            fcntl.ioctl(self.fd,0xD203,struct.pack('=2Q',cma.ctypes.data,cma.nbytes))
            fcntl.ioctl(self.fd,0xD202,struct.pack('=2Q',host.ctypes.data,host.nbytes))
        except BaseException:
            os.close(self.fd)
            self.fd=None
            raise
        self.registration_ms=(time.perf_counter()-start)*1000

    def copy(self,batched=True):
        if self.fd is None or getattr(self.cma,'freed',False):
            raise RuntimeError('DMA registration closed or CMA freed')
        request=bytearray(struct.pack('=8Q',int(batched),0,0,0,0,0,0,0))
        start=time.perf_counter()
        fcntl.ioctl(self.fd,0xD204,request,True)
        elapsed=(time.perf_counter()-start)*1000
        mode,total,sync,transfer,segments,byte_count,batches,max_inflight=struct.unpack('=8Q',request)
        if byte_count!=self.host.nbytes:
            raise RuntimeError('Incomplete DMA copy')
        return dict(batched=bool(mode),wall_ms=elapsed,kernel_ms=total/1e6,
            sync_ms=sync/1e6,transfer_wait_ms=transfer/1e6,
            segments=segments,bytes=byte_count,batches=batches,max_inflight=max_inflight)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd=None
