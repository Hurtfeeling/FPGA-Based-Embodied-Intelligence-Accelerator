/* Root-only Linux 5.4 / XLNK PS DMA experiment. No physical addresses accepted.
 * Fixed registrations own page references; caller must retain XLNK allocation.
 * Queue multiple mapped host segments before issue_pending, wait once per batch.
 * Gracefully flush if the DMA driver's descriptor pool is smaller than the SG list.
 */
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
#include <linux/sched/mm.h>

#define CONFIGURE 0xd201
#define REGISTER_HOST 0xd202
#define REGISTER_CMA 0xd203
#define COPY 0xd204
#define MAX_BYTES (32UL * 1024 * 1024)

static unsigned int batch_limit = 256;
module_param(batch_limit, uint, 0444);
MODULE_PARM_DESC(batch_limit, "Maximum queued segments per issue/wait, 1..256");

struct configuration { u64 length; u32 to_cma, channel; };
struct registration { u64 address, length; };
struct result {
	u64 mode, total_ns, sync_ns, transfer_ns, segments, bytes, batches, max_inflight;
};
struct imported {
	struct page **pages;
	unsigned int nr_pages;
	struct sg_table sgt;
	int mapped;
	bool xlnk;
	enum dma_data_direction direction;
};
struct session {
	struct mutex lock;
	struct mm_struct *owner;
	struct dma_chan *chan;
	struct device *dev;
	size_t length;
	bool to_cma;
	struct imported host, cma;
};
struct batch { struct completion done; atomic_t pending; };

static bool channel_filter(struct dma_chan *chan, void *name)
{
	return !strcmp(dma_chan_name(chan), name) && chan->device->dev->driver &&
		!strcmp(chan->device->dev->driver->name, "xilinx-zynqmp-dma");
}

static void unmap_buffer(struct session *s, struct imported *b)
{
	unsigned int i;
	if (b->mapped)
		dma_unmap_sg(s->dev, b->sgt.sgl, b->sgt.orig_nents, b->direction);
	if (b->sgt.sgl)
		sg_free_table(&b->sgt);
	for (i = 0; i < b->nr_pages; i++) {
		if (!b->xlnk && b->direction != DMA_TO_DEVICE)
			set_page_dirty_lock(b->pages[i]);
		put_page(b->pages[i]);
	}
	kfree(b->pages);
	memset(b, 0, sizeof(*b));
}

static int configure(struct session *s, unsigned long arg)
{
	struct configuration config;
	dma_cap_mask_t mask;
	const char *names[] = { "dma1chan0", "dma2chan0" };
	if (s->chan)
		return -EBUSY;
	if (copy_from_user(&config, (void __user *)arg, sizeof(config)))
		return -EFAULT;
	if (!config.length || config.length > MAX_BYTES || config.length % 64 ||
	    config.to_cma > 1 || config.channel > 1)
		return -EINVAL;
	dma_cap_zero(mask);
	dma_cap_set(DMA_MEMCPY, mask);
	s->chan = dma_request_channel(mask, channel_filter, (void *)names[config.channel]);
	if (!s->chan)
		return -EBUSY;
	s->dev = s->chan->device->dev;
	s->length = config.length;
	s->to_cma = config.to_cma;
	return 0;
}

static int register_buffer(struct session *s, unsigned long arg, bool xlnk)
{
	struct imported *b = xlnk ? &s->cma : &s->host;
	struct registration req;
	long pinned;
	int ret;
	unsigned int requested;
	if (!s->chan)
		return -EINVAL;
	if (b->pages)
		return -EBUSY;
	if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
		return -EFAULT;
	if (!req.address || req.length != s->length || req.address % 64 ||
	    req.address > ULONG_MAX - req.length)
		return -EINVAL;
	b->xlnk = xlnk;
	b->direction = (xlnk == s->to_cma) ? DMA_FROM_DEVICE : DMA_TO_DEVICE;
	requested = DIV_ROUND_UP(offset_in_page(req.address) + req.length, PAGE_SIZE);
	b->pages = kcalloc(requested, sizeof(*b->pages), GFP_KERNEL);
	if (!b->pages)
		return -ENOMEM;
	if (!xlnk) {
		pinned = get_user_pages_fast(req.address & PAGE_MASK, requested,
					     FOLL_WRITE | FOLL_LONGTERM, b->pages);
	} else {
		struct vm_area_struct *vma;
		unsigned long first = req.address & PAGE_MASK, pfn;
		unsigned int j;
		pinned = 0;
		down_read(&current->mm->mmap_sem);
		vma = find_vma(current->mm, first);
		if (!vma || vma->vm_start > first || vma->vm_end < req.address + req.length ||
		    !vma->vm_file || !S_ISCHR(file_inode(vma->vm_file)->i_mode) ||
		    strcmp(vma->vm_file->f_path.dentry->d_name.name, "xlnk")) {
			pinned = -EINVAL;
		} else {
			for (j = 0; j < requested; j++) {
				if (follow_pfn(vma, first + j * PAGE_SIZE, &pfn) || !pfn_valid(pfn))
					break;
				b->pages[j] = pfn_to_page(pfn);
				get_page(b->pages[j]);
				pinned++;
			}
		}
		up_read(&current->mm->mmap_sem);
	}
	b->nr_pages = pinned > 0 ? pinned : 0;
	if (pinned != requested) {
		ret = pinned < 0 ? pinned : -EFAULT;
		goto fail;
	}
	ret = sg_alloc_table_from_pages(&b->sgt, b->pages, requested,
				       offset_in_page(req.address), req.length, GFP_KERNEL);
	if (ret)
		goto fail;
	b->mapped = dma_map_sg(s->dev, b->sgt.sgl, b->sgt.orig_nents, b->direction);
	if (!b->mapped) { ret = -EIO; goto fail; }
	if (xlnk && (b->mapped != 1 || sg_dma_len(b->sgt.sgl) != s->length)) {
		ret = -EINVAL;
		goto fail;
	}
	dma_sync_sg_for_cpu(s->dev, b->sgt.sgl, b->sgt.orig_nents, b->direction);
	return 0;
fail:
	unmap_buffer(s, b);
	return ret;
}

static void completed(void *arg)
{
	struct batch *batch = arg;
	if (atomic_dec_and_test(&batch->pending))
		complete(&batch->done);
}

static void sync_for_device(struct session *s, struct imported *b)
{
	dma_sync_sg_for_device(s->dev, b->sgt.sgl, b->sgt.orig_nents, b->direction);
}

static void sync_for_cpu(struct session *s, struct imported *b)
{
	dma_sync_sg_for_cpu(s->dev, b->sgt.sgl, b->sgt.orig_nents, b->direction);
}

static int transfer(struct session *s, unsigned long arg)
{
	struct result result = {};
	struct scatterlist *sg;
	struct dma_async_tx_descriptor *tx;
	dma_cookie_t cookie = 0;
	dma_addr_t cma;
	u64 start, synced, finish;
	size_t offset = 0;
	unsigned int index = 0, limit;
	int error = 0;
	if (copy_from_user(&result.mode, (void __user *)arg, sizeof(result.mode)))
		return -EFAULT;
	if (!s->host.mapped || !s->cma.mapped || result.mode > 1)
		return -EINVAL;
	limit = result.mode ? batch_limit : 1;
	cma = sg_dma_address(s->cma.sgt.sgl);
	sg = s->host.sgt.sgl;
	start = ktime_get_ns();
	sync_for_device(s, &s->host);
	sync_for_device(s, &s->cma);
	synced = ktime_get_ns();
	while (index < s->host.mapped) {
		struct batch batch;
		unsigned int queued = 0;
		init_completion(&batch.done);
		/* Sentinel prevents completion while submissions are still being added. */
		atomic_set(&batch.pending, 1);
		while (queued < limit && index < s->host.mapped) {
			dma_addr_t host = sg_dma_address(sg);
			size_t length = sg_dma_len(sg);
			tx = dmaengine_prep_dma_memcpy(s->chan,
				s->to_cma ? cma + offset : host,
				s->to_cma ? host : cma + offset, length,
				DMA_PREP_INTERRUPT | DMA_CTRL_ACK);
			if (!tx)
				break; /* Descriptor pool full: issue/drain partial batch. */
			tx->callback = completed;
			tx->callback_param = &batch;
			atomic_inc(&batch.pending);
			cookie = dmaengine_submit(tx);
			if (dma_submit_error(cookie)) { error = -EIO; break; }
			queued++;
			index++;
			offset += length;
			sg = sg_next(sg);
		}
		if (error || !queued) {
			if (!error) error = -ENOMEM;
			dmaengine_terminate_sync(s->chan);
			break; /* No callback may outlive this batch's stack context. */
		}
		result.batches++;
		if (queued > result.max_inflight) result.max_inflight = queued;
		if (atomic_dec_and_test(&batch.pending))
			complete(&batch.done);
		dma_async_issue_pending(s->chan);
		if (!wait_for_completion_timeout(&batch.done, msecs_to_jiffies(5000)))
			error = -ETIMEDOUT;
		else if (dma_async_is_tx_complete(s->chan, cookie, NULL, NULL) != DMA_COMPLETE)
			error = -EIO;
		if (error) {
			dmaengine_terminate_sync(s->chan);
			break;
		}
		/* Drain callbacks and return descriptor resources before reusing context. */
		dmaengine_synchronize(s->chan);
	}
	finish = ktime_get_ns();
	sync_for_cpu(s, &s->host);
	sync_for_cpu(s, &s->cma);
	result.total_ns = ktime_get_ns() - start;
	result.sync_ns = synced - start + result.total_ns - (finish - start);
	result.transfer_ns = finish - synced;
	result.segments = s->host.mapped;
	result.bytes = offset;
	if (error)
		return error;
	return copy_to_user((void __user *)arg, &result, sizeof(result)) ? -EFAULT : 0;
}

static int session_open(struct inode *inode, struct file *file)
{
	struct session *s;
	if (!capable(CAP_SYS_RAWIO))
		return -EPERM;
	s = kzalloc(sizeof(*s), GFP_KERNEL);
	if (!s)
		return -ENOMEM;
	mutex_init(&s->lock);
	s->owner = get_task_mm(current);
	if (!s->owner) { kfree(s); return -EINVAL; }
	file->private_data = s;
	return 0;
}

static int session_release(struct inode *inode, struct file *file)
{
	struct session *s = file->private_data;
	if (s->chan) {
		dmaengine_terminate_sync(s->chan);
		unmap_buffer(s, &s->host);
		unmap_buffer(s, &s->cma);
		dma_release_channel(s->chan);
	}
	mmput(s->owner);
	kfree(s);
	return 0;
}

static long session_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct session *s = file->private_data;
	int ret;
	if (current->mm != s->owner)
		return -EPERM;
	if (mutex_lock_interruptible(&s->lock))
		return -ERESTARTSYS;
	switch (cmd) {
	case CONFIGURE: ret = configure(s, arg); break;
	case REGISTER_HOST: ret = register_buffer(s, arg, false); break;
	case REGISTER_CMA: ret = register_buffer(s, arg, true); break;
	case COPY: ret = transfer(s, arg); break;
	default: ret = -ENOTTY;
	}
	mutex_unlock(&s->lock);
	return ret;
}

static const struct file_operations operations = {
	.owner = THIS_MODULE, .open = session_open, .release = session_release,
	.unlocked_ioctl = session_ioctl, .llseek = no_llseek,
};
static struct miscdevice device = {
	.minor = MISC_DYNAMIC_MINOR, .name = "ps_dma_batch", .fops = &operations,
	.mode = 0600,
};
static int __init batch_init(void)
{
	if (!batch_limit || batch_limit > 256) return -EINVAL;
	return misc_register(&device);
}
static void __exit batch_exit(void) { misc_deregister(&device); }
module_init(batch_init);
module_exit(batch_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Isolated ZynqMP PS DMA batched host/CMA copies, up to 32 MiB");
