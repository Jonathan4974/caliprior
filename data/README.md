# Data

Download helpers for the three benchmarks. Nothing here is redistributed; each
script writes into the root configured by `setup.sh` (`TUM_ROOT`, `EUROC_ROOT`,
`KITTI_ROOT`/`KITTI_GREY_ROOT`). Run `source setup.sh` first.

```bash
bash data/download_tum.sh
bash data/download_euroc.sh
bash data/download_kitti.sh   # prints registration instructions
```

Expected layouts are documented in `docs/DATASETS.md`.
