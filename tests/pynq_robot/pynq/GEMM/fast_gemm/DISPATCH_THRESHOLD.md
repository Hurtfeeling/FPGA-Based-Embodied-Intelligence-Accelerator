# CPU / PL GEMM Dispatch

For measurements where the original M/N/K are all multiples of 32 (rather than
model inputs padded by the driver), see [the aligned-input crossover results](DISPATCH_ALIGNED32.md).

The default `cpu_threshold_ops` is `12058624`. The driver computes padded work as
`2 * round_up_32(M) * round_up_32(N) * round_up_32(K)`. A matrix at or below
the threshold uses four-thread NEON; larger matrices use GEMM_V2. In batched
inputs, each matrix is classified independently, and the remaining PL jobs are
submitted together to the available GEMM cores.

The threshold was measured on PYNQ with four CPU cores pinned. The cpufreq policy
was locked at 1199999 kHz; PYNQ reported an actual CPU clock of 1049.9895 MHz.
PL ran at 249.9975 MHz. These are actual 1.05 GHz CPU measurements, rather than
verified 1.2 GHz measurements. CPU timings include input packing. PL timings
include staging, cache synchronization, submit, wait/clear, and copying C to
ordinary host memory. Static-weight cases reuse transposed constant B after
the first call; attention cases use dynamic B and transpose it on every call.
First-use PL time is reported separately in the JSON.

Representative median timings in milliseconds:

| M x N x K | B | 4-thread NEON | PL | Faster |
|---|---|---:|---:|---|
| 50 x 720 x 128 | static | 1.659 | 1.955 | CPU |
| 50 x 720 x 256 | static | 2.536 | 1.977 | PL |
| 50 x 320 x 720 | static | 2.978 | 2.109 | PL |
| 241 x 241 x 64 | dynamic | 1.316 | 2.276 | CPU |
| 1024 x 1024 x 64 | dynamic | 13.485 | 6.490 | PL |
| 64 x 960 x 12288 | static | 237.140 | 10.029 | PL |

Three independent action-attention jobs with different dynamic B matrices at
`50 x 241 x 64` took 1.874 ms on NEON versus 7.772 ms through the grouped PL path.
For these small batches,
reducing device staging and synchronization matters more than using all three
cores.

The complete 37-case, five-repeat results, including exact-output validation
and a mixed CPU/PL dispatch check, are in
[`dispatch_threshold_results_20261103_194302.json`](dispatch_threshold_results_20261103_194302.json).
The threshold is a conservative dispatch heuristic derived from these samples;
matrix shape and B preparation can change the crossover. It is not a measured
full-model speedup. In particular, a three-slot weight cache cannot keep all
SmolVLA weights resident, so isolated static-weight cache-hit timings do not
represent full-model transpose costs. ONNX DMA integration and a full-model
measurement of this runtime version remain pending.
