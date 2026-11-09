# Data

Download helpers for the three benchmarks. Nothing here is redistributed; each
script writes into the root configured by `setup.sh` (`TUM_ROOT`, `EUROC_ROOT`,
`KITTI_ROOT`/`KITTI_GREY_ROOT`). Run `source setup.sh` first.

```bash
bash data/download_tum.sh
bash data/download_kitti.sh   # prints registration instructions
```

EuRoC datasets (`machine_hall.zip`, `vicon_room1.zip`, `vicon_room2.zip`) have to be mannually download via [EuRoc datasets](https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f) into EUROC_ROOT. Following script serves as an unzip helper.

```bash
bash data/unzip_euroc.sh
```

Expected layouts are documented in `docs/DATASETS.md`.
