# R46 frozen short/long integration comparison

Both frozen models passed the prospectively fixed64-vs128 TRAIN integration check. The sole LONG50000 candidate is closed by the unchanged joint scientific gates. No SHORT fallback was selected.

At the common64-step inference budget, extended training lowered source shape493 BA from 0.880 to 0.780, and overall PSD W1 from 0.04618358 to 0.02418821. These are real descriptive gains, while the baseline RMS W1 increased by 17.197%. LONG cannot be promoted on these marginal gains because required density support, source-conditioned PSD preservation and same-treated REAL controls also matter.

The historical R45 fixed32 recipe remains a numerical failure. R46 changes only inference budget to64 Heun steps, with128 used as a TRAIN reference;128 is not an exact solution or candidate. Frozen SHORT and LONG have2950 and50000 historical updates respectively, with94400 and1600000 Monte Carlo row appearances per domain. Their unique parent populations are unchanged. No model, optimizer, calibration, initialization, coupling or TRAIN random state was fitted in R46.

| Model | TRAIN RMS median / p95 (LSB) | preADC relative charge max | source shape493 / u480 BA | gates1–6 |
|---|---:|---:|---:|---|
| SHORT2950 | 0.047928 / 0.141050 | 1.83e-08 | 0.880 / 0.892 | T/F/T/F/F/T |
| LONG50000 | 0.105813 / 0.213042 | 1.92e-08 | 0.780 / 0.738 | T/F/F/F/F/T |

LONG minus SHORT uses the new common64-step outputs, common fixed600/250 partitions, and500 group bootstrap draws. The interval is conditional on the fitted classifiers; it does not estimate response-training or seed uncertainty.

| View | paired BA difference | 95% interval |
|---|---:|---:|
| full_historical_shape | -0.100000 | [-0.141007, -0.062014] |
| normalized_u480 | -0.154000 | [-0.196392, -0.112860] |

The prespecified complete-response improvement description is supported: **False**. Both full-shape paired intervals must have upper bounds below0 and all8 distances must be no more than10% worse. This comparison does not replace LONG-versus-BASE gates or project A–F acceptance. The observed two-view shape and PSD gains do not complete the full response-model objective.

| Distance | SHORT | LONG | LONG−SHORT relative | required missing bins |
|---|---:|---:|---:|---|
| PSD_overall_W1 | 0.04618358 | 0.02418821 | -0.4762596 | {} |
| baseline_RMS_V_overall_W1 | 4.376851e-05 | 5.12955e-05 | 0.1719728 | {} |
| ACF1_overall_W1 | 0.07183778 | 0.05451456 | -0.2411435 | {} |
| PSD_common_Q_equal_mean_W1 | 0.05942715 | 0.02478231 | -0.5829801 | {"REAL": [], "SHORT_2950_HEUN64": [], "LONG_50000_HEUN64": []} |
| component0_peak_position_MAE | 0.04313513 | undefined | undefined | {"REAL": [], "SHORT_2950_HEUN64": [], "LONG_50000_HEUN64": [2, 4]} |
| component0_raw_width_MAE | 0.01230677 | undefined | undefined | {"REAL": [], "SHORT_2950_HEUN64": [], "LONG_50000_HEUN64": [2, 4]} |
| component1_peak_position_MAE | 0.02623524 | undefined | undefined | {"REAL": [], "SHORT_2950_HEUN64": [], "LONG_50000_HEUN64": [2, 4]} |
| component1_raw_width_MAE | 0.01095356 | undefined | undefined | {"REAL": [], "SHORT_2950_HEUN64": [], "LONG_50000_HEUN64": [2, 4]} |

PSD conditional distance uses fixedQ1–7. Component center/raw width uses the original fixedQ2–7. Missing or failed required bins are retained as undefined and fail the comparison; no candidate-valid intersection is selected. Component fits are descriptive density components, not particle truth. Failed convergence or separation does not establish that physical particle peaks disappeared. All support gaps and original8-bin tables remain in metrics.json.

SHORT scientific failures: gate2_PSD_width_centers_noise, gate4_boundary_source, gate5_same_treated_REAL_control.
Required density failures: [].
Density reasons in those required bins: [].
Actual-input pure-had source bins3/4 valley changes (fraction): [-0.056338028169014086, -0.0055248618784530384].

LONG scientific failures: gate2_PSD_width_centers_noise, gate3_source_preservation, gate4_boundary_source, gate5_same_treated_REAL_control.
Required density failures: [2, 4].
Density reasons in those required bins: [{"bin_id_1based": 2, "N": 265.0, "failure_reasons": ["not_converged500", "density_components_not_separated"]}, {"bin_id_1based": 4, "N": 252.0, "failure_reasons": ["not_converged500", "density_components_not_separated"]}].
Actual-input pure-had source bins3/4 valley changes (fraction): [0.38967136150234744, 0.44198895027624313].

For the unchanged213/181 actual-input pure-had source groups in bins3/4, SHORT valley fractions are 0.000000/0.000000; LONG fractions are 0.446009/0.447514. This is a source-conditioned preservation risk, not a statement that the REAL density components are particle truth. LONG-versus-BASE_REAL baseline BA difference is 0.264000; the normalized full-shape difference is 0.028000. The fixed control gates remain unchanged.

Source2000 retains417 pureEE,1518 pure-had and65 mixed parents. Actual unquantized NN-input source-Q bins3/4 have213/181 pure-had events; historical ADC source-reference bins have207/182 and are separately labelled. All417 outside-TRAIN-support rows traverse each new model using the fixed nearest-edge conditioning index; there is no BASE fallback, clipping of source charge or reweighting. Nonzero source sensitivity does not prove correct event-wise detector mapping.

Same-treated REAL controls apply each stochastic model to the same2000 existing TRAIN-reference waveforms and original random arrays. The three gates compare new controls to frozen BASE_REAL, not to an assumed RAW identity. Absolute effects against RAW are still reported. The missing historical RAW Q–PSD2D paired reference remains unknown, without a replacement fit. REAL controls receive no G4 composition or PDG labels.

Generation covered11968 rows (8000 main,2048 repeats,1664 corrected source probes,256 source perturbations). 8000 main SOURCE+REAL rows independently replayed bitwise from original input/white through the full64-step chain and oneADC. Shared functional components retain cross covariance; they are not independent physical signal/electronics. Finite preADC charge conservation is distinct from ADC charge rounding.

Actual science cost was1696.428s; new HGB fits40, density slots32, actual EM attempts32. Server inference/evaluation/replay usedCPU12; local plots usedCPU4. Peak actual-worker RSS1933824000B; RSS probe failures1. No GPU science, new training, new REAL acquisition, EVAL or CAL waveform decoding occurred.

These are historically exposed same-acquisition development results. The short/long comparison follows one inherited optimization trajectory and unequal training costs, not independent seeds or a blinded external validation. Charge-population mismatch is preserved; no overall-spectrum acceptance rule was relaxed. Numerical success and exact replay do not establish detector-response validity.

Ordinary control failures and the pre-evaluation fixed-support correction are retained in ENGINEERING_FAILURES.json and engineering_evaluation_definition_001. The correction added no fit, generation or scientific retry. All final PNGs were actually opened; FIGURE_QA.json records the final files.

Artifacts: METRICS_COMPACT.json contains aggregate publishable evidence; full metrics/identities/random inputs/predictions/waveforms stay on E/D and the project server. GATE_METRICS.json retains each original gate and failure. REPRODUCTION.md describes the exact server64-step pipeline and8000-row replay.
