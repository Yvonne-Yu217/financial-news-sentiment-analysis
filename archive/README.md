# Archive

Materials kept for transparency but not part of the published paper artifact.

## `v3_csi_family_alternative/`

Alternative 5-index combination evaluated during R7 review cycle (2026-04-29).

**Design difference:** replaces SHCOMP / SZCOMP / CSI 300 / ChiNext / CSI 500 with a clean CSI size-decile family: SSE 50 → CSI 300 → CSI 500 → CSI 1000 → ChiNext.

**Decision: not adopted.**

| Criterion | v2 (published) | v3 alternative | Winner |
|---|---|---|---|
| G1 monotone gradient | 5/5 (forced ordering) | 5/5 (clean size-decile) | v3 |
| Data source uniformity | 3 backends | 2 backends | v3 |
| Conceptual cleanness | SZCOMP mixed-bag | pure size slicing | v3 |
| Cost to switch | — | 4-6 weeks rework | v2 |
| Submission status | v5 Accept-with-minor at PBFJ | not started | v2 |
| ChiNext effect strength | -0.00329 (NW t=-2.41) | -0.00324 (NW t=-2.47) | tie |

The full decision memo is at `v3_csi_family_alternative/v2_vs_v3_comparison.md`.

The methodological gains in v3 do not justify scrapping v2's submission-ready state.
A future stock-level extension (referenced in §6 Limitations) is the appropriate place
to adopt the cleaner CSI family indices.
