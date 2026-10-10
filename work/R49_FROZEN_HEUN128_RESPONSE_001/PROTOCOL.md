# R49 implementation

The registered R49 task is authoritative. Frozen R48 models and calibration are reused.
Only common candidate Heun128/reference Heun256 changes; CPU12/chunk64.
Both original128 TRAIN rows must pass before any DEV access. Original gates and A-C unchanged.
All2000 SOURCE rows processed, including417 outside TRAIN support; conditioning clamp only.
11968 generated rows,40 fixed HGB designs,32 density slots,8000 independent main replay rows.
Fixed Q1-7 PSD and Q2-7 density support; missing bins cannot be deleted.
R48 historical50000-step training per arm is referenced; R49 training/calibration=0.
Time/deadline/output quota fields are null. Machine RSS/single-job safety remains.
CPU work does not require idle GPU. Native elapsed and unknown observations remain recorded.
No migration equivalence tests. E/D local persistence. No writes to frozen older stages.
