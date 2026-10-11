# Candidate and evidence

Only `_flatten_gradients_cpu_f64` needs a root-owned substitution. The candidate
packs homogeneous float32/float64 gradients in original parameter order on their
current device, transfers native dtype once, casts to CPU float64, then checks
finiteness on CPU. Mixed devices/dtypes and other dtypes use the old sequence.
No chunk fusion, GPU gradient accumulation, optimizer change, RNG change, or
projection change is proposed. Original per-64-row CPU float64 addition and the
single final cast/clip=5/Adam call remain unchanged.

Actual synthetic ResponseNetwork has 174 parameter tensors and 1,369,889 scalar
parameters. For the 104 D chunks, this changes 18,096 per-parameter finite-check
host observations and D2H calls to 104 CPU checks and native-dtype D2H calls.
This is an operation-count observation, not a measured CUDA speedup. One packed
float32 device buffer is 5,479,556 bytes (about 5.23 MiB). Every chunk still has
one synchronization-producing D2H transfer.

`CPU_ARTIFICIAL_RECEIPT.json` passed: exact bytes include signed zero, float32 and
float64 subnormals, noncontiguous gradients, mixed-dtype fallback, NaN/Inf and
metadata rejection, original64 chunk addition, D then C, unchanged clip/Adam
parameters and state for three artificial updates, and RNG preservation. Native
CPU batch=1 endpoint/replay/VJP uses newly constructed artificial weights with a
nonzero final head; all 174 gradient tensors are nonzero. Peak RSS=353,181,696
bytes; affinity mask=2, one numerical thread, hard private-commit cap=1 GiB.
This does not establish CUDA equivalence or scientific acceptance.

Root CUDA entry:

```text
python -B benchmark_cuda_artificial.py --source-root <frozen-R65-directory> --output <review-directory>/CUDA_ARTIFICIAL_RECEIPT.json --threads 1 --samples 7 --pack-repetitions 20 --max-seconds 150
```

Copy `gradient_pack_candidate.py`, `artificial_common.py`, and the benchmark
entry together. The script tests old `.to(cpu, float64)` versus separate native
D2H/CPU cast, including subnormals/signed zero, mixed dtype/device fallback, and
actual 64-row artificial endpoint/VJP. It reports packing timing and complete
64-row forward+VJP timing separately, alternating order with synchronization.
No optimizer step is run. Root owns training pause/resume and resource guards.

Deployment gate: all CUDA byte comparisons must pass, and root must assess the
actual64 forward+VJP gain rather than packing-only gain. Failed equivalence or
no useful gain means retaining the original run. No full-step or 50,000-step ETA
follows from this bounded artificial benchmark.

Remaining limits: the finite check moves after the native transfer/CPU cast, so
nonfinite failures are detected at the same chunk but later within packing. If
several malformed gradients coexist, exception ordering may differ. Mixed-device
fallback is prepared for CUDA testing and not exercised by CPU-only execution.
CUDA conversion/subnormal behavior and server-library behavior remain unverified
until root executes the prepared benchmark. No scientific arrays, saved weights,
or optimizer state were read; no frozen files, Git state, registrations, global
metadata, training process, or remote system were modified by this agent.
