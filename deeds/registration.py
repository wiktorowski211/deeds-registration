import os
import tempfile
import subprocess

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from .utils import *


def registration(fixed, moving, verbose=True):
    libs_path = pkg_resources.files('deeds.libs')

    deeds_path = os.path.join(libs_path, "deedsBCV.so")

    stdout = None if verbose else subprocess.DEVNULL

    # TODO: Use in-memory communication instead of the filesystem
    with tempfile.TemporaryDirectory() as tmp_dir:
        fixed_path = os.path.join(tmp_dir, "fixed.nii.gz")
        moving_path = os.path.join(tmp_dir, "moving.nii.gz")
        moved_path = os.path.join(tmp_dir, "moved")

        save_nifty(fixed, fixed_path)
        save_nifty(moving, moving_path)

        command = f"{deeds_path} -F {fixed_path} -M {moving_path} -O {moved_path}"

        subprocess.run(command.split(), stdout=stdout)

        moved_path = f"{moved_path}_deformed.nii.gz"

        moved = load_nifty(moved_path)

        return moved
