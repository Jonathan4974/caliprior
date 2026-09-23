"""Path resolution for the CaliPrior release.

Every location is overridable through an environment variable. When a variable
is unset, a sensible default is derived from this file's location, so the
release runs unchanged when the recommended layout is used (see `setup.sh` and
`docs/SETUP.md`).

Placeholders / variables
------------------------
PROJECT_ROOT   this release's root (default: parent of this `code/` directory)
HISLAM2_ROOT   checkout of the HI-SLAM2 backend used in the paper
CONDA_ROOT     conda installation holding the `hislam2` environment
VENV_ROOT      CPU virtualenv used for cache building / analysis
DATA_ROOT      directory that contains the dataset mirrors
EUROC_ROOT     EuRoC MAV root          (default: $DATA_ROOT/euroc)
TUM_ROOT       TUM RGB-D root          (default: $DATA_ROOT/tum_rgbd_benchmark)
KITTI_ROOT     KITTI odometry color    (default: $DATA_ROOT/kitti_odom_color/...)
KITTI_GREY_ROOT KITTI odometry grey    (default: $DATA_ROOT/kitti_odom_grey/sequences)
HF_HOME        Hugging Face cache root (default: ~/.cache/huggingface)
"""
import os

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env(name, default=""):
    return os.environ.get(name, default)


PROJECT_ROOT = _env("PROJECT_ROOT", os.path.dirname(_CODE_DIR))
HISLAM2_ROOT = _env("HISLAM2_ROOT", "")
CONDA_ROOT = _env("CONDA_ROOT", "")
VENV_ROOT = _env("VENV_ROOT", "")
DATA_ROOT = _env("DATA_ROOT", "")

EUROC_ROOT = _env("EUROC_ROOT", os.path.join(DATA_ROOT, "euroc"))
TUM_ROOT = _env("TUM_ROOT", os.path.join(DATA_ROOT, "tum_rgbd_benchmark"))
KITTI_ROOT = _env(
    "KITTI_ROOT",
    os.path.join(DATA_ROOT, "kitti_odom_color", "data_odometry_color", "dataset"),
)
KITTI_GREY_ROOT = _env(
    "KITTI_GREY_ROOT", os.path.join(DATA_ROOT, "kitti_odom_grey", "sequences")
)

HF_HOME = _env("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
