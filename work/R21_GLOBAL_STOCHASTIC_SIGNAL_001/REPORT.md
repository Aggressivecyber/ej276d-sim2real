# R21: global stochastic signal response

GLOBAL_STYLE produces measurable stochastic signal changes and restores the prespecified middle-charge pure-had centers relative to the saved R20/FINE result. Baseline ACF1 W1 falls from 0.338123 to 0.215922, and Q/PSD HGB BA falls from 0.844 to 0.788. The complete response remains unsuccessful: full shape BA is 0.992, all temporal/local voltage views remain highly separable, and both density-component conditional widths remain too narrow. Different hardware and update counts limit attribution to the new package.

This stage keeps the complete G4-derived 480-point source waveform and all original 2000 DEV parents. It adds one eight-dimensional style modulation plus latent-recovery package to the R19/TILT starting translator, while reusing the R20/FINE frozen source sampling, fine-time losses and actual old random streams. Signal and electronic labels identify network branches; neither is established physical noise truth or a conditional expectation.

Local CPU execution completed 2699 G/D/reader updates; selected step 2000. J20 initial/selected/final = 0.75490725 / 0.61663079 / 0.64895546. Validation latent L1 initial/selected/final = 0.79968679 / 0.77937722 / 0.72970074. Model selection used only fixed internal TRAIN validation. Latent error reduction alone is not response realism.

The saved R20/FINE reference used 4000 GPU updates with different Torch/NumPy/sklearn versions. The new local completed old-stream prefix matches are {"source": true, "target": true, "bin": true, "source_uniform": true, "NCE_positions": true, "noise_stream": true}. Full equal-update matching is False. Initial J20 difference from that historical environment = 1.1920929e-07. This is an environment/update-limited historical comparison, not a strictly isolated equal-compute ablation.

## Complete waveform and PSD

| Fixed HGB view | R20/FINE BA | GLOBAL_STYLE BA |
|---|---:|---:|
| full_historical_shape | 0.996 | 0.992 |
| full_voltage | 0.986 | 0.988 |
| prompt_local | 0.988 | 0.986 |
| postpeak_local | 0.986 | 0.994 |
| tail_local | 0.982 | 0.992 |
| baseline_local | 0.996 | 0.998 |
| Q_PSD_2D | 0.844 | 0.788 |


The classifiers use the same saved DEV holdout parents/artificial REAL groups. Paired bootstrap intervals condition on the fitted classifiers and do not include response-training or acquisition uncertainty. A score near one retains a large domain gap; a lower training loss or attractive mean curve does not establish success.

| Domain | overall PSD W1 | mean of all 8 supported PSD W1 | baseline ACF1 W1 |
|---|---:|---:|---:|
| SOURCE_CLEAN | 0.10330028 | 0.10602419 | undefined |
| R19_TILT | 0.039168306 | 0.045835205 | 0.27961508 |
| R20_FINE | 0.036347363 | 0.044430713 | 0.33812323 |
| GLOBAL_STYLE | 0.036959508 | 0.043239201 | 0.21592178 |


Every eight-bin count, center/raw width/detrended width and failure remains in METRICS_COMPACT. Density components are descriptions, not incoming-particle ground truth. Width errors only use the same mutually qualified bins; unsupported bins cannot be counted as passing. The old q4 reference nonconvergence remains unchanged; no fallback or refit repairs it.

On bins1-4, jointly qualified across REAL/SOURCE_CLEAN/R19_TILT/R20_FINE/GLOBAL_STYLE, upper-component center MAE is 0.009086 versus FINE 0.028157, and raw-sigma MAE is 0.038965 versus 0.042385. Lower-component raw-sigma MAE is 0.018849 versus 0.018497, so width improvement is not uniform. GLOBAL_STYLE and both historical translated references fail bin8 two-component qualification; REAL remains qualified in all8. The Q/PSD BA paired delta is -0.056, 95% cluster-bootstrap interval [-0.091159,-0.022025]; full-shape delta -0.004 has interval [-0.013918,0.005849]. These are fixed-classifier descriptive comparisons, not an acquisition-independent equivalence test.

## Prespecified pure-had source bins

| Domain | bin 3 output PSD mean | bin 3 valley fraction | bin 4 output PSD mean | bin 4 valley fraction |
|---|---:|---:|---:|---:|
| SOURCE_CLEAN | 0.37405854 | 0 | 0.3745278 | 0 |
| R19_TILT | 0.27333153 | 0 | 0.27214798 | 0 |
| R20_FINE | 0.24064882 | 0.68115942 | 0.24748627 | 0.23076923 |
| GLOBAL_STYLE | 0.26846594 | 0 | 0.26971185 | 0.0054945055 |


These are source-charge-conditioned pure-deposition groups, not a classification of REAL or incoming PDG. Output-conditioned bins, original source mixture, PDG inference and all source/output charge migration are reported separately. Source conditioning is required to avoid hiding displacement by output-charge selection.

## Actual random response and source controls


The original 128 parents ×8 joint draws plus 32 fixed parents ×32 style-only draws and the same 32×32 electronic-only draws total 3072 outputs. First decomposition draws share an actual joint reference. Variance uses n−1; waveform RMS SD is the square root of mean point variance. Repeated outputs do not increase independent G4 parents.

| Mode | composition | parents | mean PSD sample SD | output RMS SD [mV] | signal RMS SD [mV] | electronic RMS SD [mV] | parents straddling frozen valley |
|---|---|---:|---:|---:|---:|---:|---:|
| joint8 | mixed | 42 | 0.0064685065 | 1.6843856 | 1.4888095 | 0.8639613 | 0 |
| joint8 | pure_EE | 42 | 0.0069649337 | 1.4003919 | 1.1619502 | 0.81902051 | 0 |
| joint8 | pure_had | 44 | 0.0076261215 | 1.6812375 | 1.4916318 | 0.86601188 | 0 |
| style_only32 | pure_EE | 16 | 0.006881948 | 1.1100007 | 1.1398241 | 0.17982454 | 0 |
| style_only32 | pure_had | 16 | 0.0074350945 | 1.5580512 | 1.6004483 | 0.22121282 | 0 |
| electronic_only32 | pure_EE | 16 | 0.0018860781 | 0.80094863 | 0 | 0.80082872 | 0 |
| electronic_only32 | pure_had | 16 | 0.0019627537 | 0.84225944 | 0 | 0.84198935 | 0 |


Style affects the signal branch and can indirectly change the old electronic branch through modulated features. Random variation or recoverable latent codes may still be texture or class splitting. Conversely, zero same-parent valley straddling does not validate the pure-group centers. There is no repeated-measurement REAL truth for these same G4 parents.

The existing corrected 832 source probes keep actual source/noise/reference parents distinct. Fixed-source changes are measured relative to that fixed source reference. Same-input/same-all-noise bitwise check = True. Nonzero source sensitivity is not calibrated physical correspondence. The new modulation gradient was finite and nonzero; 96 pre-update structural rows passed and were reused after the indexing-only engineering repair.

## Execution, failures and reproducibility


Two terminal engineering failures occurred before successful training: frozen-table schema KeyError (G/D 0/0; 11.858320s) and uint16 NCE index incompatibility (G/D 0/1; 10.956510s). Their source/log/native artifacts remain in failed_001 and failed_002. The fixes preserve lambda values and actual NCE positions. One failed D update and the completed structural rows remain charged; the new update cap is 3999 and training time deducts the second failure's conservative wall. Numerical cumulative wall including saved-result title QA = 650.864700s; monitored main/replay/failed-worker peak RSS = 929660928 bytes; CPU4/GPU0.

A later saved-PSD-only rendering clarified the gray valley caption from ambiguous own-Q to source-Q, retaining the original image. Its figure completed correctly, but an incomplete Windows memory-counter structure made the post-render RSS check fail (native1). The original receipt and 2.3245094s conservative tool wall remain charged; RSS for that small helper is unknown. It performed no model generation, feature computation or fit and was not rerun to repair the instrumentation.

Independent saved-model replay used only the selected translator and actual source/style/z/white: 2000 rows, 960000 points, pre/post ADC and parent identities bitwise equal. No latent reader, REAL/TRAIN or fitting was used. All execution/cache/data writes stay E/D. SSH, installation, EVAL/new G4/raw-long decoding and new CAL-waveform decoding are zero. Config retains historical server Python/old absolute-deadline metadata; actual local native receipt and R21 local addendum determine execution.

REAL comes from one acquisition; groups of 50 are artificial bookkeeping, not independent runs. This historically exposed DEV diagnostic is not an external blind test. G4 clean waveforms encode total EE/had deposition via the existing analytic recipe, not microscopic deposition timing. The learned unpaired mapping and soft source sampling assumptions do not identify a unique per-event detector response.

Necessary weights, optimizer/end states, actual inputs/streams, full metrics, decomposed waveforms, source controls and classifier predictions remain in this local stage. REPRODUCTION.md gives the entry and required existing inputs; FIGURE_QA.json records actual visual inspection. The next scientific direction is for ROOT to register after reviewing the residuals; no extra arm, sweep or fit was added.
