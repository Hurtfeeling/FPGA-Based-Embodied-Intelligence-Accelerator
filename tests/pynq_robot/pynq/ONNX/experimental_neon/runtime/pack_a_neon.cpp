#include <arm_neon.h>
#include <omp.h>
#include <sched.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

extern "C" int pack_a_u8_to_s8(const uint8_t* src, int8_t* dst,
                                int rows, int k, int padded_rows,
                                int padded_k, int src_row_stride,
                                int threads, const int* cpu_ids) {
    if (!src || !dst || !cpu_ids || rows <= 0 || k <= 0 ||
        padded_rows < rows || padded_k < k || src_row_stride < k ||
        (threads != 1 && threads != 4)) return -1;

    int error = 0;
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
#pragma omp parallel num_threads(threads) reduction(max:error)
    {
        const int tid = omp_get_thread_num();
        const int parts = omp_get_num_threads();
        cpu_set_t old_mask;
        CPU_ZERO(&old_mask);
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu_ids[tid], &mask);
        if (parts != threads || sched_getaffinity(0, sizeof(old_mask), &old_mask) ||
            sched_setaffinity(0, sizeof(mask), &mask)) {
            error = 1;
        }

        const int first = tid * padded_rows / parts;
        const int last = (tid + 1) * padded_rows / parts;
        const uint8x16_t offset = vdupq_n_u8(128);
        for (int r = first; r < last; ++r) {
            int8_t* out = dst + static_cast<size_t>(r) * padded_k;
            if (r >= rows) {
                std::memset(out, 0, padded_k);
                continue;
            }
            const uint8_t* in = src + static_cast<size_t>(r) * src_row_stride;
            int col = 0;
            for (; col + 16 <= k; col += 16) {
                const uint8x16_t q = vld1q_u8(in + col);
                const uint8x16_t s = vsubq_u8(q, offset);
                vst1q_s8(out + col, vreinterpretq_s8_u8(s));
            }
            for (; col < k; ++col)
                out[col] = static_cast<int8_t>(static_cast<uint8_t>(in[col] - 128));
            if (k < padded_k) std::memset(out + k, 0, padded_k - k);
        }
        if (sched_setaffinity(0, sizeof(old_mask), &old_mask)) error = 1;
    }
    omp_set_dynamic(old_dynamic);
    return error ? -1 : 0;
}
