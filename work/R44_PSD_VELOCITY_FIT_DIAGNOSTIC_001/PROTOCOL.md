# R44 frozen path diagnostic

One frozen R43 final model, eval/no_grad, chunk32, exact94400 consumed rows and saved float32 t. Reconstruct original white in the original z/white call order and original ordinal; assert hashes and final RNG. No optimizer, training, ODE, generation, HGB, EM, bootstrap, new data or input selection.

q=sum100:400-3sum0:100; F=sum100:150-.5sum0:100. Unit-charge PSD=1-F. For standardized velocity u, L_b(u)=-F(s_b*u). Recover a and c with identical R43 float64 arithmetic; network is float32 and projected field float64. Compare L(c-a) against PSD(c_unit)-PSD(a_unit); max1e-10. Preserve charge-tangent residual max1e-10.

g=-s*c_fast; h=s*c_charge; gT=g-h*(g.h)/(h.h). Residual e=pred-target is decomposed into its gT direction and orthogonal complement. Energies are sums in standardized coordinate squared velocity; direction share is ratio of sums, not mean row ratios. PSD velocity units are dimensionless per unit time; MSE is standardized coordinate squared velocity. Fixed8 bins and4 time intervals; empty groups undefined.

Population background is descriptive and separate: actual consumed and unique REAL targets, complete REAL TRAIN, original RAW REAL DEV, frozen R43 preADC/postADC output, actual preADC SOURCE. SOURCE/output bins fixed by actual SOURCE preADC Q; REAL by its own preADC Q, with outside support retained. Plot complete empirical CDFs without clipping/resampling. Do not infer particle truth or physical pairs.

Direction marker in original bins2-7: abs(B)/D>=.20 and B/G same sign, B=mean signed PSD-velocity residual, D=mean abs target PSD velocity, G=mean saved R43 main preADC PSD minus consumed-target PSD. D0 undefined. At least4/6 only supports a hypothesis; fewer closes this explanation. No new response automatically authorized.

Each chunk is written to a scalar memmap and flushed; progress saves attempted/committed rows. A failed/over-budget run preserves its prefix, never silently reruns the whole forward. Final results distinguish94400 exposures,3752/18739 unique consumed records and6593/21047 loaded pools.

Registered acceptance is prospective after R43, does not retrospectively amend prior science. This900-second diagnostic belongs to6300-second improvement block with at most two later response endpoints. LocalE/D only, noSSH/installation/agents/Git. Wait only the current root resource update before the sole forward; definition is frozen.

## Resource addendum before any actual forward

R44_SERVER_RESOURCE_ADDENDUM_001 and SERVER_RESOURCE_POLICY v2 override local-only/caps/deadlines: unique server CUDA chunk32 forward, CPU12, treeRSS48GiB, GPU allocated/reserved/process10GiB, science3600s, outputs2GiB; executor2026-10-10T03:39:27.880232Z. Saved scalar rows return to E for sequential CPU4/RSS4GiB population and plotting. Device change is disclosed, no cross-device bitwise assertion; all mathematics and direction markers unchanged. Original6300-second resource block is superseded by RESOURCE_BUDGET v2; scientific acceptance unchanged.
