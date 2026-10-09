# R14 TRAIN-pool expansion

Expanded TRAIN has 21047 QC-valid records from21100 raw candidates, compared with original2592. The original599 validation and2000 DEV are unchanged. This is one acquisition with artificial groups, not an increase in independent acquisitions. The task changes network sample size while the FACTOR representation and empirical residual L remain estimated on original2592.

Increasing TRAIN improves complete validation likelihood and reduces the train–validation gap from about55–61 to about5–8 nat/event. At the same2000 updates, full shapeBA is0.928 GLOBAL and0.926 CONDITIONAL versus small0.900/0.932. After16300 updates it is0.918/0.910. These remain far above REAL–REAL0.512, while full voltage remains0.970–0.988. More data helps statistical generalization but does not solve the complete response discrepancy. Increased compute is separated from sample-size change.

## Selection and execution

Exactly18500 new TRAIN raw rows were streamed and decoded; 18455 passed inherited QC, 45 failed and were not replaced. Original2592 cached valid rows remain first, new valid rows append by ascending original_index. TRAIN_POOL434 groups minus exactly12 existing val groups gives422 candidate fit groups. No DEV/CAL/EVAL row or group enters training; CAL numeric decode/EVAL/newG4 are zero. Compressed/decompressed bytes may traverse unselected raw rows for CRC/EOF without numeric decode. No whole-container hash pass or migration content test occurred.

The executed source uses the unchanged R4 stream and raw20-point mean,14-bit ADC, float32 baseline/peak,CFD480-window conversion. Extraction actual cost and QC bits are in EXTRACTION_RECEIPT.json. The remote supervisor owns one child at a time, extraction then two-network training/evaluation then fresh inference replay.

## Continuous training and two selection budgets

Each network starts from actual R13 saved initial parameters/permutations, with original seeds, batch128, Adam lr0.0003/weightdecay1e-5,clip5. Same initial state does not imply the same minibatch sequence when N changes. No R13 trained checkpoint, restart, schedule or DEV-based model selection is used.

| Network | Actual updates / U | <=2000 selected step / valNLL | <=U selected step / valNLL | Final valNLL |
|---|---:|---:|---:|---:|
| head | 16300/16300 | 2000 / 32.193184 | 15700 / 31.109238 | 31.169050 |
| tail | 16300/16300 | 2000 / -2303.721191 | 15900 / -2306.726318 | -2306.605957 |

U=100 ceil(2000*N/2592/100), capped16300. Small mean draws/record=98.765432; large at2000=12.163254; actual exposure counts and completed-budget flags are retained for each network. Equal-updates changes sample size at fixed maximum updates; equal-exposures additionally increases computation, so that difference cannot be attributed only to more data. Step0 and every100 are the only checkpoint grid, earliest exact tie.

## Complete validation and generated distributions

| Arm | Full599 val NLL[nat/event] | Expanded TRAIN NLL | Val−TRAIN gap | Shape BA | Voltage BA | Tail BA |
|---|---:|---:|---:|---:|---:|---:|
| UPDATES_GLOBAL | -2373.758840 | -2379.977317 | 6.218477 | 0.928 | 0.982 | 0.91 |
| UPDATES_CONDITIONAL | -2416.525722 | -2421.316711 | 4.790989 | 0.926 | 0.988 | 0.9359999999999999 |
| EXPOSURES_GLOBAL | -2374.842793 | -2382.082307 | 7.239514 | 0.918 | 0.98 | 0.9259999999999999 |
| EXPOSURES_CONDITIONAL | -2420.614998 | -2428.807348 | 8.192350 | 0.9099999999999999 | 0.97 | 0.9299999999999999 |
| SMALL_GLOBAL | reused original selected model / unchanged val | see original-fit score below | — | 0.9 | 0.98 | 0.908 |
| SMALL_CONDITIONAL | reused original selected model / unchanged val | see original-fit score below | — | 0.932 | 0.98 | 0.914 |
| REAL_REAL | reused original selected model / unchanged val | see original-fit score below | — | 0.512 | 0.546 | 0.49 |

Scores use saved selected weights in CPUfloat64 and the full480-dimensional Jacobian. Complete all-fit NLL and original R13 selected-weight original2592-fit score are report-only; they do not change checkpoint selection. Original small train/val gaps:

```json
{
  "SMALL_GLOBAL": {
    "TRAIN_N": 2592,
    "val_N": 599,
    "TRAIN_NLL": -2433.259133596948,
    "val_NLL": -2371.9174377407353,
    "val_minus_TRAIN_NLL": 61.34169585621248
  },
  "SMALL_CONDITIONAL": {
    "TRAIN_N": 2592,
    "val_N": 599,
    "TRAIN_NLL": -2453.8532889838507,
    "val_NLL": -2398.435901393619,
    "val_minus_TRAIN_NLL": 55.417387590231556
  }
}
```

## Predeclared paired effects

All six candidates share original R7 grouped600/250 pools and their common QC support. REAL–REAL is a same-N sampling reference, not an equivalence threshold. Every view is in full/compact metrics. Shape BA differences (first minus second):

| Contrast | Delta BA | Paired95%CI |
|---|---:|---|
| UPDATES_GLOBAL_minus_SMALL_GLOBAL | 0.028000 | [0.006147646650416025, 0.050618480173994805] |
| UPDATES_CONDITIONAL_minus_SMALL_CONDITIONAL | -0.006000 | [-0.03391228788361119, 0.02204393600779893] |
| EXPOSURES_GLOBAL_minus_UPDATES_GLOBAL | -0.010000 | [-0.03511712691537342, 0.012668234938737437] |
| EXPOSURES_CONDITIONAL_minus_UPDATES_CONDITIONAL | -0.016000 | [-0.039351135190918464, 0.005425125457402538] |
| UPDATES_CONDITIONAL_minus_UPDATES_GLOBAL | -0.002000 | [-0.025664572438334646, 0.02066115702479338] |
| EXPOSURES_CONDITIONAL_minus_EXPOSURES_GLOBAL | -0.008000 | [-0.03831674900612665, 0.02350346766545809] |

These500 artificial-group bootstrap intervals are conditional on fitted classifiers, not retraining or independent-acquisition uncertainty; no corrected familywise-significance/equivalence claim is made. Within each budget the two tail arms share exactly the same computed head and tail innovations. Across budgets pairing is actual draw position, not the same physical head or G4 event. Every new arm retains2000 outputs without resampling, q/PSD clamp or extra CFD.

## Conditional support and memory limits

| Domain | QC valid/invalid | Qualified narrow-bin IDs1..8 |
|---|---:|---|
| REAL | 2000/0 | [1, 2, 3, 4, 5, 6, 7, 8] |
| SMALL_GLOBAL | 2000/0 | [7] |
| SMALL_CONDITIONAL | 2000/0 | [7] |
| UPDATES_GLOBAL | 2000/0 | [] |
| UPDATES_CONDITIONAL | 2000/0 | [] |
| EXPOSURES_GLOBAL | 2000/0 | [1, 2, 6] |
| EXPOSURES_CONDITIONAL | 2000/0 | [1, 2, 6] |

Existing local timing statistics (no added calculation):

| Domain | cross50 mean / sampleSD[index] | cross50 q05 / median / q95 | peak mean / sampleSD[index] |
|---|---|---|---|
| REAL | 119.475478 / 0.281736 | [119.05707562004437, 119.44150280629032, 119.93844977002651] | 126.270500 / 2.109397 |
| SMALL_GLOBAL | 119.506014 / 0.521973 | [118.91303043902583, 119.51088348890627, 120.12828812702782] | 126.362000 / 2.369127 |
| SMALL_CONDITIONAL | 119.498410 / 0.471482 | [118.90950375755618, 119.50934201774481, 120.11958870498347] | 126.344000 / 2.382331 |
| UPDATES_GLOBAL | 119.525759 / 0.547397 | [118.95311061968862, 119.50171399658502, 120.09089057753343] | 126.393000 / 2.355276 |
| UPDATES_CONDITIONAL | 119.521256 / 0.549354 | [118.95121464671091, 119.50095139415228, 120.08730653351196] | 126.409500 / 2.375633 |
| EXPOSURES_GLOBAL | 119.501325 / 0.491332 | [118.99478061604856, 119.48854277799256, 120.03562508450477] | 126.291500 / 2.319961 |
| EXPOSURES_CONDITIONAL | 119.492145 / 0.492408 | [118.96016719660454, 119.48905013472894, 120.02900968451476] | 126.297500 / 2.298616 |

REAL cross50 full range118.9995–120.0049, while EXPOSURES_CONDITIONAL114.4951–127.1213. Larger TRAIN does not enforce the narrow observed alignment support; this is a descriptive residual, not proof of a physical cause. Prompt100:150 curvature contains the peak/early fall and is not exclusively an upstroke statistic.

Six-way common-support records:

```json
[
  {
    "scope": "six_way",
    "component": 0,
    "field": "mean",
    "common_bin_ids_1based": [],
    "MAE": {
      "SMALL_GLOBAL": null,
      "SMALL_CONDITIONAL": null,
      "UPDATES_GLOBAL": null,
      "UPDATES_CONDITIONAL": null,
      "EXPOSURES_GLOBAL": null,
      "EXPOSURES_CONDITIONAL": null
    }
  },
  {
    "scope": "six_way",
    "component": 0,
    "field": "sigma_PSD",
    "common_bin_ids_1based": [],
    "MAE": {
      "SMALL_GLOBAL": null,
      "SMALL_CONDITIONAL": null,
      "UPDATES_GLOBAL": null,
      "UPDATES_CONDITIONAL": null,
      "EXPOSURES_GLOBAL": null,
      "EXPOSURES_CONDITIONAL": null
    }
  },
  {
    "scope": "six_way",
    "component": 0,
    "field": "detrended_sigma_PSD",
    "common_bin_ids_1based": [],
    "MAE": {
      "SMALL_GLOBAL": null,
      "SMALL_CONDITIONAL": null,
      "UPDATES_GLOBAL": null,
      "UPDATES_CONDITIONAL": null,
      "EXPOSURES_GLOBAL": null,
      "EXPOSURES_CONDITIONAL": null
    }
  },
  {
    "scope": "six_way",
    "component": 1,
    "field": "mean",
    "common_bin_ids_1based": [],
    "MAE": {
      "SMALL_GLOBAL": null,
      "SMALL_CONDITIONAL": null,
      "UPDATES_GLOBAL": null,
      "UPDATES_CONDITIONAL": null,
      "EXPOSURES_GLOBAL": null,
      "EXPOSURES_CONDITIONAL": null
    }
  },
  {
    "scope": "six_way",
    "component": 1,
    "field": "sigma_PSD",
    "common_bin_ids_1based": [],
    "MAE": {
      "SMALL_GLOBAL": null,
      "SMALL_CONDITIONAL": null,
      "UPDATES_GLOBAL": null,
      "UPDATES_CONDITIONAL": null,
      "EXPOSURES_GLOBAL": null,
      "EXPOSURES_CONDITIONAL": null
    }
  },
  {
    "scope": "six_way",
    "component": 1,
    "field": "detrended_sigma_PSD",
    "common_bin_ids_1based": [],
    "MAE": {
      "SMALL_GLOBAL": null,
      "SMALL_CONDITIONAL": null,
      "UPDATES_GLOBAL": null,
      "UPDATES_CONDITIONAL": null,
      "EXPOSURES_GLOBAL": null,
      "EXPOSURES_CONDITIONAL": null
    }
  }
]
```

All unsupported bins/failure reasons remain visible. No empty common support is replaced by different-bin averages or marginalPSD success. The inherited q4 reference failure and q1 HC-neutron absence remain; fitted density components/high-confidence classes are not particle truth.

All8000 new outputs were queried against original2592 and expanded21047 TRAIN references, in V and own-shape views. Their source/undefined counts and exact-copy counts are kept. Different reference cardinality changes nearest distances; a reduction does not itself show memorization. No exact match is not proof of no memory. Historical REAL/R13 summaries are reused rather than newly fitted.

## Replay, resources and scope

Independent replay PID167065 status=pass; all8000 rows/3840000 points use actual saved weights/innovations. Per-arm premax/tolerance and postarray_equal are in REPLAY_RECEIPT.json. No fit/new draws/full evaluation replay or repeated Jacobian audit.

Numeric cumulative=200.645504s /2400, including extraction, computation/figures and replay; network training=149.352518s /1200. Peak RSS=2913755136B /4GiB; GPU allocated peak=116336128B /4GiB. HGB=42/42, widthslots=32/32, EM=32/32. Actual update/time/termination histories and all failures are preserved.

Eight delivered figures were actually viewed. Mean waveform proximity does not establish complete distributional equality. Larger REAL cost belongs in paper A accounting. Even a target-density improvement would not establish G4-to-REAL response or downstream task utility; original independent-label/run limitations remain. No new scientific task is started by closure.

## Reproduction and artifacts

See REPRODUCTION.md, config.json/PROTOCOL.md, SELECTION.json/DATA_ROLES.json, TRAIN_ADDITION.npz, selected/final/initial network states and traces, four generated.npz files, metrics.json/METRICS_COMPACT.json, native/replay/resource receipts, figures, and one FINAL_MANIFEST.json. Private event/index/prediction/NN payloads remain local and are excluded from compact publication metrics.
