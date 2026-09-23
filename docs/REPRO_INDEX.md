# Reproducibility index

Mapping from each result in the paper to its script and output file. Outputs are
under `results/` in this release (originally `outputs/` in the working tree).
Exact commands for every row are in `docs/RUNS.md`; the end-to-end order is in
`scripts/run_pipeline.sh`.

| result | script | output |
|---|---|---|
| gauge blind spot | `gauge_null.py` | `gauge_null.json` |
| audit / in-domain fix | `run_stock.py`, `run_wired.py --gauge ma` | `{stock,wired,gaugeMA}_*` |
| gauge law | `run_wired.py` (gauge sweep) | `{gaugeB,anchorA}_*` |
| free-signal candidate | `observable_b.py`, `gauge.py` | `{observable_b,gauge_map}.json` |
| five-arm ladder | `run_gauge_ladder.py` | `gauge_ladder.json` |
| trust gate | `gate_ood.py` | `gate_ood.json` |
| depth-mean anchor | `build_anchor.py` | `anchors.json`, `anchorN_*` |
| trajectory anchor | `build_traj_anchor.py` | `traj_anchor.json` (16 seq, 13 accepted) |
| ten-sequence gate + protocol | `gate_ood.py`, `run_protocol.py` | `{gate_ood,protocol}.json` |
| DA-V2 model | `build_dav2_cache.py` | `dav2_*`, `gate_ood_dav2.json` |
| local drift | `local_drift.py` | `local_drift*.json` |
| gate robustness | `gate_robust.py` | `gate_robust{,dav2,da3m}.json`, `gate_robust_16.json` |
| protocol in metres | `run_protocol_ate.py` | `protocol_ate.json`, `protocol_ate_16.json` |
| low-parallax certificate | `gauge_parallax.py`, `lowparallax.py` | `{gauge_parallax,lowparallax}.json` |
| DA3METRIC model | `build_da3m_cache.py` | `da3m_*`, `gate_ood_da3m.json` |
| depth-mean anchor, all ten | `build_anchor.py`, `summarize_anchors.py` | `{anchors,anchor_compare}.json`, `anchorN2_*` |
| KITTI control | `run_stock.py`, `run_wired.py --gauge ma` | `{stock,wired}_kitti00_800/` |
| metric-factor arms | `jdsa_modes.py --mode metric`, `summarize_e1.py` | `e1_{stock,metricconst,metricconf,metricadopt}_*` |
| alpha sweep + local step | `run_wired.py` (metric mode) | `e1_metric*_a*/`, `e1_metricconf_step_*` |
| sigma ladder + regime | `run_wired.py --sigma-adapt/--sigma-regime`, `regime_nis.py` | `e2_*`, `e2reg_*`, `regime_nis.txt` |
| regime K-sensitivity + guard | `guard_compare.py` | `e2regG_*` |
| per-model gate calibration | `gate_model_calib.py` | `gate_model_calib.json` |
| gate blind spot | `build_ma_cache.py --k-scale`, `gate_diag.py` | `ma_cache_k120/`, `e3_k120_gaugeMA_*`, `gate_diag.txt` |
| gate expansion, 16 seq | `build_ma_cache.py --no-gs` | `ma_cache/`, `noprior_*`, `gaugeMA_*`, `gate_ood_16.json` |
| anchor/protocol at 16 seq | `build_traj_anchor.py --seqs`, `run_protocol_ate.py --seqs`, `gate_robust.py` | `traj_anchor.json`, `protocol_ate_16.json`, `gate_robust_16.json` |
| paper figures | `make_figs.py`, `make_figs2.py`, `make_figs3.py` | `figures/caliprior_{summary,tier12,mechanisms}.png` |

## Corrections applied to the draft

1. The confidence-weighted arms were re-run after fixing a keying misalignment
   between the served confidence tile and the backend's keyframe indexing;
   every confidence number in the paper is post-fix.
2. The earlier "local lever is live" claim is withdrawn: it came from an
   undocumented statistic and does not reproduce under the paper's GT-based tool.
3. The per-regime sigma result is reported as fragile, not supported, after its
   pre-registered K-sensitivity failed.
