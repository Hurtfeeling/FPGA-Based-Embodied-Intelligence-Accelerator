/* GEMM_V2: volatile 32-bit AXI-Lite accesses, sequential configure/START.
 * Only receives live PYNQ MMIO virtual mappings from the owning Python driver.
 * No /dev/mem access, physical-address mapping, allocation or clock changes.
 */
#include <stdint.h>

static inline void barrier(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

void fg_submit(const uintptr_t *bases, const uint32_t *values, unsigned count) {
    unsigned i, j;
    for (i = 0; i < count; ++i) {
        volatile uint32_t *r = (volatile uint32_t *)bases[i];
        for (j = 0; j < 5; ++j) r[1 + j] = values[5 * i + j];
        barrier(); /* Parameters visible before START. */
        r[0] = 0x101;
        barrier(); /* START core i before configuring core i+1. */
    }
}

/* Only poll cores that are not already drained. Status encodes:
 * 0=pending, 1=success, 2=drained hardware error, 3=unexpected task ID.
 * No performance registers are accessed by this function.
 */
void fg_poll(const uintptr_t *bases, const uint32_t *jobs, unsigned count,
             uint32_t skip_mask, uint32_t *states) {
    unsigned i;
    for (i = 0; i < count; ++i) {
        volatile uint32_t *r = (volatile uint32_t *)bases[i];
        uint32_t flags;
        if (skip_mask & (1u << i)) continue;
        flags = r[0x18 / 4];
        barrier();
        if (flags & 65) { states[i] = 0; continue; }
        if (flags & 4) { states[i] = 2; continue; }
        if (!(flags & 128)) { states[i] = 0; continue; }
        states[i] = (r[0x28 / 4] == jobs[i] && r[0x20 / 4] == jobs[i]) ? 1 : 3;
        barrier();
    }
}

/* Called only after every participating core is confirmed drained. */
void fg_clear(const uintptr_t *bases, unsigned count, uint32_t *flags,
              uint32_t *accepted) {
    unsigned i;
    for (i = 0; i < count; ++i) {
        volatile uint32_t *r = (volatile uint32_t *)bases[i];
        r[0] = 0x201;
        barrier();
        flags[i] = r[0x18 / 4];
        accepted[i] = r[0x20 / 4];
        barrier();
    }
}
