# 08 — Compute and cost notes

<a id="k"></a>
Measured numbers that may save others time. Hardware: rented Colab GPUs (a T4 with 15 GB, later an L4 with 23 GB and 12 CPU cores) driven through a job queue, and a 4-core CPU machine without GPU.
Every figure is from our own job logs unless marked. No account details are given; an L4 cost about 1.9 compute units per hour at the time.
"Per pass" = one checkpoint, one layer order, one depth window.

## K1 — Detector throughput

| detector, setting | T4 | L4 | CPU |
|---|---|---|---|
| reader-v3, fp16, 384 px tiles, 50 % overlap | 350–800 cm²/h per pass (typically 600–730) | 480–790 cm²/h per pass | about 77 cm²/h on 3 threads when idle; 24–33 under load |
| `ink_9um`, fp16, 128 px tiles | 630–770 cm²/h per pass | not measured | 22–44 cm²/h on 2 threads |
| Hecate 9.6 µm, fp16 | 41–70 cm²/h per pass | 134 cm²/h per pass | 0.34 cm²/h |
| Hecate 9.6 µm, bf16 / fp32 | 12 / 18 cm²/h | not measured | — |
| Studio, stride 96, fp16, batch 16 | about 17 s per cm² per order (210 cm²/h) at 8.64 µm | about 10–12 s per cm² per order (350 cm²/h) | 0.96 s per window with bf16 and channels-last on 3 threads; about 26 cm²/h idle, 9–13 under load |
| Studio, stride 48 with blending, fp16 | about 52 s per cm² per pass at 9.37 µm | 41–47 s per cm² per order | about 4× the stride-96 time |
| canonical 2.4 µm, stride 128, fp16, batch 4 | about 20 cm²/h per order | 34 cm²/h per pass | not run |
| Hecate 2.4 µm, stride 128 | not run | 34 cm²/h per pass | not run |
| either 2.4 µm model, stride 192 | not run | about 74 cm²/h per pass | not run |

- reader-v3 on a GPU is bound by CPU-side tiling: only about 17 % of its wall time was forward passes in one measurement, and more CPU threads did not help in another.
- On small masked patches reader-v3's rate drops (12.5 cm²/h on CPU for a 0.37 cm² patch): tile overhang dominates.
- Standard four passes at 2.4 µm (two models, two orders): about 8.3 cm² per L4-hour, 7.2 minutes per cm².

## K2 — Studio memory and precision

| setting | per window | peak allocated |
|---|---|---|
| T4, batch 1, fp32 | 340 ms | 11.35 GB (fits, barely) |
| T4, batch 1, fp16 | 132 ms | 1.17 GB |
| T4, batch 16, fp16 | 105 ms | 5.7 GB |
| T4, batch 32, fp16 | 105 ms | 10.6 GB (14.6 GB reserved) |
| L4, batch 4, fp32 | 96 ms | 15.7 GB |
| L4, batch 16, fp16 | 83 ms | 7.6 GB (9.2 GB reserved) |
| L4, batch 32, fp16 | 88 ms | 14.3 GB |

- A T4 has no native bf16; the release's documented default should be replaced by fp16 there.
- fp16 differs from fp32 by at most 0.003 in probability; bf16 by up to 0.035. Prefer fp16.
- Under a memory cap of about 5 GB, batch 4–8 in fp16 fits (2.1–3.3 GB allocated on a T4).

## K3 — Surfaces

| step | time | data |
|---|---|---|
| normal grids for one scroll, xy band of ±2,200 slices | 456–493 s on 8 cores; 740 s in one fresh run | 9–10 GB |
| normal grids, whole xz/yz plus two xy bands | 1,350–1,930 s per scroll, four downloads in parallel | 7–9 GB; the xz/yz grids cannot be cut to a region |
| patch growing, 70 / 80 / 100 / 130 generations | about 3 min / 350 s / 650–880 s / 1,042 s | chart 5.6–6.6 / 7.5 / 11.7–13.7 / 23 cm² |
| render, 41 layers, 9 µm | 15–30 s per cm²; 44 s for a 7.7 cm² window on 8 cores | about 0.35 GB of chunks per cm² |
| render, 109 layers, 2.4 µm | 10.9 process-minutes per cm²; about 25 cm²/h wall with 6 processes | 10.6 GB read per cm²; 1.9 GB of stack per cm² |
| m7 on-sheet check of a window | 36–44 s plus a 74–120 s m7 render | — |
| snap to m7 | 25 s for 3 windings; 117 s for 120 windings on 6 processes | m7 level 0 band streamed |
| atlas winding: download, check, prepare | about 15 s per cm² of CPU | stacks deleted after each winding |
| label-free pass over an atlas winding | about 2.5 s per cm² including download | peak disk about 1 GB |
| first window of a scroll, end to end, 8-core T4 machine | about 40 min (about 1 h on 2 cores) | grids dominate |
| a tranche window, GPU part (reader-v3 two orders × two shifts, Studio two orders at stride 96) | 9–11 min on an L4, 12–14 min on a T4 | — |

## K4 — Spiral fits

| item | PHerc. 0358 (T4) | PHerc. 0175A (L4) | PHerc. 1203 (L4) | PHerc. 0846A (L4) |
|---|---|---|---|---|
| band height (slices) | 1,500, later 1,000 | 1,500 | 1,400 | 1,300 |
| tracks in band / points | 1.90 M / 86.1 M | 2.01 M / 145.5 M | 1.24 M / 53.7 M | 1.49 M / 80.2 M |
| data preparation (CPU) | 1,479 s | 1,621 s | about 14 min plus track extraction | about 15 min plus track extraction |
| of which the surface distance field | 950 s, 8.0 GB | 676 s, 8.7 GB | 626 s, 4.2 GB | 427 s |
| track extraction (CPU) | published store | published store | 1,529 s on 8 workers | 1,397 s on 11 workers |
| synthetic normals from m7 (GPU) | not needed | 28 s for the computation | not needed | under 5 min for the job |
| peak GPU memory | 12.0–12.4 GB (10.4–10.5 GB for 1,000 slices) | 19.8 GB | 9.5–9.8 GB | 12.4–12.6 GB |
| seconds per step | 1.07 (0.89 for 1,000 slices) | 0.5–0.6 | about 0.55 | similar to PHerc. 1203 |
| stall before the first step of every run | about 7.5 min (track resampling, not cached) | 8.5 min, shorter on resume | present | present |
| 30,000 steps | not run; about 9 hours estimated | about 8 L4-hours with gates | 5.3 L4-hours | running |
| checkpoint size | not measured | 686 MB | 649 MB | not recorded |
| gate on one checkpoint (CPU: export, snap, check, four renders) | — | 1,347 s | 1,236 s | — |

- Environment: the fitter's `uv sync` as documented picks a Python version without a wheel for one locked dependency; `uv sync --python 3.14` works. Setup on a fresh machine takes about 1.5 minutes.
- Memory scales with the band: resident pools on the GPU were normals 1.0–1.8 GiB plus distance field 2.6–6.4 GiB.
- Chunks of 26–35 minutes with a clean stop and resume survived every interruption; the one fit without saved checkpoints (PHerc. 0358, 7,000 steps) lost its final meshes when the machine stopped.

## K5 — GPU time spent per lane

| lane | GPU time |
|---|---|
| first scan of 18 renders (59.6 cm²), reader-v3 and `ink_9um`, 10 passes each | 3.1 T4-hours; 2.2–2.8 T4-minutes per cm² all-in |
| Studio feasibility and 11 windows, with follow-up | 3.8 + 0.75 T4-hours |
| atlas, PHerc. 0813 and 0800 (884 cm² valid, two passes) with positives | 5.6 T4-hours |
| atlas, PHerc. 0125 and small scrolls, Studio subsets, tile tests | 2.1 L4-hours, plus about 2 L4-hours of reader-v3 inks |
| work-up of F1 (base rate, extension windows, strokeness sweep) | 4.3 L4-hours |
| own-window tranche (22 inked windows, controls, follow-ups) | 2.65 L4-hours and 2.5 T4-hours; stride-48 follow-ups and the F2 work-up about 1.8 L4-hours more |
| spiral fits on PHerc. 0358 | 4.1 T4-hours |
| ink on the PHerc. 0175A windings (520 cm², four reader-v3 passes; Studio on 50 cm²; one work-up) | 6.4 L4-hours |
| 2.4 µm control, sweep and 39 blocks of PHerc. 1203 | 2.7 L4-hours |
| PHerc. Paris 4 title lane | about 4.4 L4-hours plus 1.6 hours on a shared session |

## K6 — Things that cost us time

- A rented machine can be replaced without warning; everything not copied out is lost. We lost prepared stacks, float maps and one fit's meshes this way. Copy checkpoints and float maps out of the machine in every chunk.
- Several lanes share one GPU through a fair-share queue; wall-clock time was often three times the GPU time.
- A 403 or any status other than 200/404 from the bucket must stop a run. Only 404 means "no chunk here"; treating other errors as empty chunks silently turns papyrus into blank.
- Deleting stacks after post-processing saved disk and made later re-review impossible: for PHerc. 0813 and 0800 only montages and half-resolution maps survive.
- Batch-size 32 for Studio is slower than 16 and doubles memory.
