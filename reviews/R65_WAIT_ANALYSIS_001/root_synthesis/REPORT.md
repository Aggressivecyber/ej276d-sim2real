# R65 waiting analysis: work accounting and GAP dependency

This is static preparation during the single active R65 training run. It does
not change its method, source population, stochastic streams, 50000-update
endpoint or original A-F. No scientific arrays or weights were decoded.

## Full empirical work and observed timing

Each update includes two 6593-parent neural passes plus the original 384-row C
pass: 13570 neural rows and 209 forward calls. It uses 104 full-D parameter VJPs,
one C reverse pass, one combined gradient installation, one clip and one Adam
step. Across the fixed 50000 updates the committed accounting is 678500000
neural rows, 10450000 forward calls and 5200000 full-D VJPs. These are repeated
operations, not distinct training parents or extra independent measurements.
The full-D SOURCE and REAL empirical laws respectively contribute 329650000
and 1052350000 repeated rows. Each step still has only 6593 and 21047 unique
parents. Artificial checks, failed attempts and replayed work remain separate.

The two saved intervals at the same full-D parameter-replay phase average
42.082 and
6.565 seconds per update.
These are observed operational intervals, not a benchmark, a cause diagnosis,
or a reliable completion-date estimate. The variation is why neither a single
instantaneous GPU utilization value nor the sparse 100-step training log should
be treated as a stalled worker. No profiling workload or runtime change was made.

## Conditional empty-region consequence for the future GAP study

The published R3B eight-bin definition puts Q6 in
[4849.0997550341,6435.57228640361) mV ns. That whole interval lies strictly inside
the original CAL q3 exclusion [3703.77829484642,6711.804904975) mV ns. Q6 and CAL q3
are different partition labels; their numerical endpoints must be compared.

The saved source chain uses the R3B width definition in R23 data_support, saves
it as TRAIN_PRIVATE.edges, carries it into the R33 affine constants and R42
unit-charge constants, then copies those constants into R43. R65 loads the
frozen R43 constant archive. The current full-population R63 receipt reports
399 SOURCE and 2624 REAL parents in raw region 5, the zero-based Q6 category.
Those are FULL counts from existing metadata, not newly filtered GAP counts.

If the GAP exclusion uses exactly the same signed saved-window charge and those
bound edges, Q6 must be empty after exclusion, independently of the individual
waveforms. This is a conditional consequence of interval containment and saved
source/metadata lineage. We did not decode the live constants or filter any
actual parents here. The registered preflight must still verify actual edge
bytes, source nominal-clean charge versus training-input charge, REAL charge,
parent identity and every remaining region count. It must not silently replace
the source's original GAP charge definition with a different measured quantity.

The unchanged R65 trainer requires all ten SOURCE and REAL regional pools to be
nonempty; ParentLayout rejects an empty regional list and the original C step
draws 32 occurrences per region. Refitting unit-charge moments from an empty Q6
REAL pool is also undefined. Therefore the future GAP preparation should first
test this concrete applicability dependency. It cannot merely remove a region,
reuse FULL moments, or alter regional mass/thresholds and claim the same recipe.
No GAP job is registered or executed, and no part of E is completed by this note.

The active FULL training continues unchanged. This issue affects the design and
interpretation of a conditional future scope study, not the current A-C verdict,
which is still unavailable. It does not establish that the original model goal
is impossible or authorize a change in the current scientific endpoint.
