// Isolated, root-only DMAEngine experiment for Linux 5.4 / ZynqMP.
// Uses one exclusively requested PS channel and owned coherent memory only.
// Never accepts physical addresses. Imported XLNK memory must stay allocated
// until this device is closed; the Python wrapper owns it through that point.
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/dmaengine.h>
#include <linux/dma-mapping.h>
#include <linux/completion.h>
#include <linux/mm.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/ktime.h>
#include <linux/capability.h>

#define BUF_BYTES (4UL * 1024 * 1024)
#define REGISTER_HOST 0xd101
#define HOST_TO_DMA 0xd102
#define DMA_TO_HOST 0xd103
#define REGISTER_CMA 0xd104
#define SET_DIRECTION 0xd105

struct trial_request { u64 address, length; };
struct trial_result { u64 total_ns, sync_ns, transfer_ns, segments, bytes; };
struct trial {
	struct dma_chan *chan;
	struct device *dev;
	void *buffer;
	dma_addr_t dma;
	struct page **pages;
	unsigned int nr_pages;
	struct sg_table sgt;
	int mapped;
	struct mutex lock;
	struct trial *external;
	enum dma_data_direction direction;
};

static bool channel_filter(struct dma_chan *chan, void *unused)
{
	return !strcmp(dma_chan_name(chan), "dma1chan0") &&
		chan->device->dev->driver &&
		!strcmp(chan->device->dev->driver->name, "xilinx-zynqmp-dma");
}

static void unmap_host(struct trial *t)
{
	unsigned int i;
	if (t->mapped) {
		dma_unmap_sg(t->dev, t->sgt.sgl, t->sgt.orig_nents, t->direction);
		t->mapped = 0;
	}
	if (t->sgt.sgl) {
		sg_free_table(&t->sgt);
		memset(&t->sgt, 0, sizeof(t->sgt));
	}
	for (i = 0; i < t->nr_pages; i++) {
		set_page_dirty_lock(t->pages[i]);
		put_page(t->pages[i]);
	}
	kfree(t->pages);
	t->pages = NULL;
	t->nr_pages = 0;
}

static int trial_open(struct inode *inode, struct file *file)
{
	struct trial *t;
	dma_cap_mask_t mask;
	if (!capable(CAP_SYS_RAWIO))
		return -EPERM;
	t = kzalloc(sizeof(*t), GFP_KERNEL);
	if (!t)
		return -ENOMEM;
	mutex_init(&t->lock);
	t->direction = DMA_BIDIRECTIONAL;
	dma_cap_zero(mask);
	dma_cap_set(DMA_MEMCPY, mask);
	t->chan = dma_request_channel(mask, channel_filter, NULL);
	if (!t->chan) {
		kfree(t);
		return -EBUSY;
	}
	t->dev = t->chan->device->dev;
	t->buffer = dma_alloc_coherent(t->dev, BUF_BYTES, &t->dma, GFP_KERNEL);
	if (!t->buffer) {
		dma_release_channel(t->chan);
		kfree(t);
		return -ENOMEM;
	}
	file->private_data = t;
	return 0;
}

static int trial_release(struct inode *inode, struct file *file)
{
	struct trial *t = file->private_data;
	dmaengine_terminate_sync(t->chan);
	if (t->external) {
		unmap_host(t->external);
		kfree(t->external);
	}
	unmap_host(t);
	dma_free_coherent(t->dev, BUF_BYTES, t->buffer, t->dma);
	dma_release_channel(t->chan);
	kfree(t);
	return 0;
}

static int trial_mmap(struct file *file, struct vm_area_struct *vma)
{
	struct trial *t = file->private_data;
	if (vma->vm_pgoff || vma->vm_end - vma->vm_start != BUF_BYTES)
		return -EINVAL;
	return dma_mmap_coherent(t->dev, vma, t->buffer, t->dma, BUF_BYTES);
}

static int register_host(struct trial *t, unsigned long arg, bool xlnk)
{
	struct trial_request req;
	long pinned;
	int ret;
	if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
		return -EFAULT;
	if (req.length != BUF_BYTES || !req.address ||
	    req.address > ULONG_MAX - BUF_BYTES)
		return -EINVAL;
	unmap_host(t);
	t->nr_pages = DIV_ROUND_UP(offset_in_page(req.address) + BUF_BYTES, PAGE_SIZE);
	t->pages = kcalloc(t->nr_pages, sizeof(*t->pages), GFP_KERNEL);
	if (!t->pages) {
		t->nr_pages = 0;
		return -ENOMEM;
	}
	if (!xlnk) {
		pinned = get_user_pages_fast(req.address & PAGE_MASK, t->nr_pages,
					     FOLL_WRITE | FOLL_LONGTERM, t->pages);
	} else {
		struct vm_area_struct *vma;
		unsigned long first = req.address & PAGE_MASK;
		unsigned long pfn;
		unsigned int j;
		pinned = 0;
		down_read(&current->mm->mmap_sem);
		vma = find_vma(current->mm, first);
		// Import only this process's actual /dev/xlnk mapping, never /dev/mem.
		if (!vma || vma->vm_start > first ||
		    vma->vm_end < req.address + BUF_BYTES || !vma->vm_file ||
		    !S_ISCHR(file_inode(vma->vm_file)->i_mode) ||
		    strcmp(vma->vm_file->f_path.dentry->d_name.name, "xlnk")) {
			pinned = -EINVAL;
		} else {
			for (j = 0; j < t->nr_pages; j++) {
				if (follow_pfn(vma, first + j * PAGE_SIZE, &pfn) || !pfn_valid(pfn))
					break;
				t->pages[j] = pfn_to_page(pfn);
				get_page(t->pages[j]);
				pinned++;
			}
		}
		up_read(&current->mm->mmap_sem);
	}
	if (pinned != t->nr_pages) {
		t->nr_pages = pinned > 0 ? pinned : 0;
		unmap_host(t);
		return pinned < 0 ? pinned : -EFAULT;
	}
	ret = sg_alloc_table_from_pages(&t->sgt, t->pages, t->nr_pages,
				       offset_in_page(req.address), BUF_BYTES, GFP_KERNEL);
	if (ret) {
		unmap_host(t);
		return ret;
	}
	t->mapped = dma_map_sg(t->dev, t->sgt.sgl, t->sgt.orig_nents, t->direction);
	if (!t->mapped) {
		unmap_host(t);
		return -EIO;
	}
	dma_sync_sg_for_cpu(t->dev, t->sgt.sgl, t->sgt.orig_nents, t->direction);
	return 0;
}

static int register_cma(struct trial *t, unsigned long arg)
{
	struct trial *ext;
	int ret;
	if (t->external)
		return -EBUSY;
	ext = kzalloc(sizeof(*ext), GFP_KERNEL);
	if (!ext)
		return -ENOMEM;
	ext->dev = t->dev;
	ext->direction = t->direction == DMA_TO_DEVICE ? DMA_FROM_DEVICE :
		(t->direction == DMA_FROM_DEVICE ? DMA_TO_DEVICE : DMA_BIDIRECTIONAL);
	ret = register_host(ext, arg, true);
	if (!ret && (ext->mapped != 1 || sg_dma_len(ext->sgt.sgl) != BUF_BYTES))
		ret = -EINVAL;
	if (ret) {
		unmap_host(ext);
		kfree(ext);
		return ret;
	}
	t->external = ext;
	return 0;
}

static void completed(void *arg)
{
	complete(arg);
}

static int transfer(struct trial *t, bool to_dma, unsigned long arg)
{
	struct scatterlist *sg;
	struct dma_async_tx_descriptor *tx;
	struct trial_result result = {};
	DECLARE_COMPLETION_ONSTACK(done);
	dma_cookie_t cookie;
	u64 start, synced, finish;
	size_t offset = 0;
	dma_addr_t target = t->external ? sg_dma_address(t->external->sgt.sgl) : t->dma;
	int i, error = 0;
	if (!t->mapped)
		return -EINVAL;
	if ((t->direction == DMA_TO_DEVICE && !to_dma) ||
	    (t->direction == DMA_FROM_DEVICE && to_dma))
		return -EINVAL;
	start = ktime_get_ns();
	dma_sync_sg_for_device(t->dev, t->sgt.sgl, t->sgt.orig_nents, t->direction);
	if (t->external)
		dma_sync_sg_for_device(t->dev, t->external->sgt.sgl,
				       t->external->sgt.orig_nents, t->external->direction);
	synced = ktime_get_ns();
	// Submit and wait one segment at a time: conservative correctness baseline.
	// Both descriptor setup and interrupt/wait overhead are included in timing.
	for_each_sg(t->sgt.sgl, sg, t->mapped, i) {
		dma_addr_t host = sg_dma_address(sg);
		size_t len = sg_dma_len(sg);
		reinit_completion(&done);
		tx = dmaengine_prep_dma_memcpy(t->chan,
			to_dma ? target + offset : host,
			to_dma ? host : target + offset, len,
			DMA_PREP_INTERRUPT | DMA_CTRL_ACK);
		if (!tx) { error = -ENOMEM; break; }
		tx->callback = completed;
		tx->callback_param = &done;
		cookie = dmaengine_submit(tx);
		if (dma_submit_error(cookie)) { error = -EIO; break; }
		dma_async_issue_pending(t->chan);
		if (!wait_for_completion_timeout(&done, msecs_to_jiffies(5000))) {
			error = -ETIMEDOUT;
			break;
		}
		if (dma_async_is_tx_complete(t->chan, cookie, NULL, NULL) != DMA_COMPLETE) {
			error = -EIO;
			break;
		}
		offset += len;
	}
	// Drain callbacks before the on-stack completion or any mappings disappear.
	if (error)
		dmaengine_terminate_sync(t->chan);
	else
		dmaengine_synchronize(t->chan);
	finish = ktime_get_ns();
	dma_sync_sg_for_cpu(t->dev, t->sgt.sgl, t->sgt.orig_nents, t->direction);
	if (t->external)
		dma_sync_sg_for_cpu(t->dev, t->external->sgt.sgl,
				    t->external->sgt.orig_nents, t->external->direction);
	result.total_ns = ktime_get_ns() - start;
	result.sync_ns = synced - start + result.total_ns - (finish - start);
	result.transfer_ns = finish - synced;
	result.segments = t->mapped;
	result.bytes = offset;
	if (error)
		return error;
	return copy_to_user((void __user *)arg, &result, sizeof(result)) ? -EFAULT : 0;
}

static long trial_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct trial *t = file->private_data;
	int ret;
	if (mutex_lock_interruptible(&t->lock))
		return -ERESTARTSYS;
	switch (cmd) {
	case REGISTER_HOST: ret = register_host(t, arg, false); break;
	case REGISTER_CMA: ret = register_cma(t, arg); break;
	case SET_DIRECTION:
		if (t->pages || t->external || arg > 1) {
			ret = -EINVAL;
		} else {
			t->direction = arg ? DMA_TO_DEVICE : DMA_FROM_DEVICE;
			ret = 0;
		}
		break;
	case HOST_TO_DMA: ret = transfer(t, true, arg); break;
	case DMA_TO_HOST: ret = transfer(t, false, arg); break;
	default: ret = -ENOTTY;
	}
	mutex_unlock(&t->lock);
	return ret;
}

static const struct file_operations trial_fops = {
	.owner = THIS_MODULE,
	.open = trial_open,
	.release = trial_release,
	.mmap = trial_mmap,
	.unlocked_ioctl = trial_ioctl,
	.llseek = no_llseek,
};

static struct miscdevice trial_device = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "ps_dma_trial",
	.fops = &trial_fops,
	.mode = 0600,
};

static int __init trial_init(void) { return misc_register(&trial_device); }
static void __exit trial_exit(void) { misc_deregister(&trial_device); }
module_init(trial_init);
module_exit(trial_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Isolated ZynqMP PS DMA ordinary user memory copy experiment");
