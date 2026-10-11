# R68: finite TRAIN charge compatibility before more training

The R67 same-start/equal-time comparison did not establish a joint benefit for full-distribution training. R65 remains retired and held for cost. This separate diagnostic reads only the existing R64 TRAIN charge scalars; it neither retries that pilot nor starts a candidate model. Original A-F, source weights and all closed negative results remain unchanged.

## Fixed input and endpoint

Read exactly SOURCE_original_q (6593 parents) and REAL_canonical_q (21047 parents) from work/R64_FROZEN_REFERENCE_GRADIENT_DIAGNOSTIC_001/ENDPOINT_GEOMETRY_PRIVATE.npz. These values were written by endpoint_geometry.py:375-382 from the complete original TRAIN populations. The original SOURCE scalar was calculated from the float32 input using q=sum(v[100:400])-3*sum(v[0:100]); its units are V samples. Preserve all parents equally, including support-outside rows. Bind the input archive and source code hashes before decoding either scalar. No waveform, model weight, DEV, EVAL or CAL decode; no model inference, training, density fit, classifier fit, randomness or parameter sweep.

Perform one exact finite-input analysis with the two prespecified descriptions: raw scalar KS and the conservative interval bound below. Every possible threshold is a deterministic part of computing the bound, not model selection. There is no held-out claim, confidence interval or estimated population law.

## Admissible class and conservative ADC interval

For each stored input q_i>0, allow output charge in [q_i-d_i,q_i+d_i], with d_i=q_i/100000+600/16383+1/10000000. This interval is a superset of outputs obeying the current per-parent preADC relative-charge condition, the bounded [-0.5,0.5] float32 preADC representation, and the current float32 ADC followed by canonical-code conversion. It relaxes the actual discrete ADC output, making the bound conservative.

The coefficient L1 norm of q is 100*3+300=600. In canonical_adc_torch, float32 addition and scaling before rounding incur at most 2^-24 V and 2^-11 codes; integer rounding adds half a code. Hence the first ADC code has error no larger than 0.5/16383+2^-24+2^-11/16383, strictly less than one LSB. Float32 division and subtraction in the legacy decode change its value by at most 2^-24+2^-25 V; recoding this value recovers the same integer because the resulting code error is below half a code. Float64 decoding/reduction error is covered by the additional 1e-7 V-sample margin. This margin also exceeds the worst-case float64 reduction/comparison error for 400 bounded preADC coordinates; it does not relax the original gate. No new clipping is admitted. The bound is conditional on the current representation and operator, not a universal result for arbitrary future ADC implementations.

The original numerical check compares each output with its original input q, not N-step output with 2N-step output. A rank-preserving charge quantile map cannot be silently substituted. Nevertheless the final A-F evaluates the original 2000 SOURCE parents and a fixed HGB; requiring the declared charge condition on this entire TRAIN population is an explicit class assumption of this diagnostic and is not a new acceptance requirement.

## Exact calculation and interpretation

Represent each stored float64 scalar by its exact binary rational. Construct interval endpoints with exact rational arithmetic. At every unique lower endpoint, upper endpoint and REAL charge, compute L(t)=#(upper_i<=t)/6593, U(t)=#(lower_i<=t)/6593 and F_R(t)=#(q_R<=t)/21047. Any admissible output CDF satisfies L<=F_out<=U, so KS(F_out,F_R) is at least max_t(max(L-F_R,F_R-U,0)). Integer cross-products determine the maximum; no floating tolerance decides the result. Right-continuous CDF values on this complete finite breakpoint set cover every interval, including left-limit values represented by preceding intervals. Save the maximizing threshold/direction and counts, and independently recount it with a direct loop.

With balanced domain classes, a corresponding charge threshold has BA at least (1+bound)/2 on these same finite parent laws. Report whether the exact bound exceeds 0.20, equivalent to this descriptive threshold BA exceeding 0.60. A positive bound excludes exact equality of the joint Q-PSD finite laws within the stated class. It does NOT prove that the original fixed-HGB held-out BA confidence interval must fail, that the original 2000 DEV parents have this bound, or that original A-F is universally impossible. Threshold choice used the full TRAIN scalar populations; it is not a fitted-and-tested HGB result or independent acquisition evidence.

If the robust gap is material, do not spend more on the premise that longer charge-preserving training can erase this population difference. Preserve the original acceptance and require any next mechanism to state which residual it can change and how a capped test would justify its cost. Neither outcome restarts R65, permits population reweighting, or establishes a successful model.

## Execution and delivery

One local worker, one CPU, native hard working-set maximum 1 GiB; local collective limits remain CPU4/RSS4GiB. One scientific computation at a time. A supervisor stops this worker at 60 seconds; it has a 50-second internal reserve. Five exact artificial cases cover identical laws, unequal laws, both witness directions, endpoint ties and overlapping allowed intervals. No retry or additional scientific endpoint under this registration. Report actual native exit, wall time, decoded fields, input/output hashes, root review and a rendered CDF figure inspected directly. E/D only for every new local file. Git receives code and compact evidence; private arrays stay in E/D backup. A-F remains active and incomplete.
