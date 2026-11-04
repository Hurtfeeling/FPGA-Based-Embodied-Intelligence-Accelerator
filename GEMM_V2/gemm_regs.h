#ifndef GEMM_REGS_H
#define GEMM_REGS_H
/* ABI 10 field reference; software examples use PYNQ Python. */
#define GEMM_ABI_VERSION 10u
#define GEMM_CTRL                 0x00u
#define GEMM_A_BASE               0x04u
#define GEMM_BT_BASE              0x08u
#define GEMM_Y_BASE               0x0Cu
#define GEMM_DIM_MN               0x10u
#define GEMM_DIM_K                0x14u
#define GEMM_FLAGS                0x18u
#define GEMM_TILES_DONE           0x1Cu
#define GEMM_ACCEPTED_JOB         0x20u
#define GEMM_COMPUTE_DONE_JOB     0x24u
#define GEMM_WRITE_DONE_JOB       0x28u
#define GEMM_WRITER_JOB           0x2Cu
#define GEMM_WRITER_MASKS         0x30u
#define GEMM_CYCLES_LO            0x34u
#define GEMM_CYCLES_HI            0x38u
#define GEMM_COMPUTE_CYCLES       0x3Cu
#define GEMM_LOAD_CYCLES          0x40u
#define GEMM_STORE_CYCLES         0x44u
#define GEMM_READ_BYTES           0x48u
#define GEMM_WRITE_BYTES          0x4Cu
#define GEMM_CAP_CONFIG           0x50u
#define GEMM_STATUS GEMM_FLAGS
#define GEMM_REGISTER_BYTES 0x54u
#define GEMM_CAP_CONFIG_VALUE 0x0A080020u
#define GEMM_ENABLE (1u<<0)
#define GEMM_IRQ_ENABLE (1u<<1)
#define GEMM_START (1u<<8)
#define GEMM_CLEAR_STATUS (1u<<9)
#define GEMM_BUSY (1u<<0)
#define GEMM_COMPUTE_DONE (1u<<1)
#define GEMM_ERROR (1u<<2)
#define GEMM_PERF_SAT (1u<<4)
#define GEMM_WRITE_BUSY (1u<<6)
#define GEMM_WRITE_DONE (1u<<7)
#define GEMM_MAX_K 16352u
#define GEMM_M32_MASK 0xFFFFu
#define GEMM_N32_SHIFT 16u
#define GEMM_K32_MASK 0xFFFFu
#define GEMM_EFFECTIVE_TM32_SHIFT 8u
#define GEMM_EFFECTIVE_TN32_SHIFT 12u
#define GEMM_EFFECTIVE_TK32_SHIFT 16u
#endif
