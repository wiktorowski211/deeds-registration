from libcpp cimport bool

import SimpleITK as sitk
import numpy as np

cimport numpy as np

cdef extern from "libs/deedsBCV0.h":
    void deeds(float *im1, float *im1b, float *warped1, int m, int n, int o, float alpha, int levels, bool verbose)


def deeds_cpp(np.ndarray[np.float32_t, ndim=1] fixed,
              np.ndarray[np.float32_t, ndim=1] moving,
              np.ndarray[np.float32_t, ndim=1] moved,
              shape, alpha, level, verbose):
    return deeds(&moving[0], &fixed[0], &moved[0],
                  shape[2], shape[1], shape[0],
                  alpha, level, verbose)


def registration(fixed, moving, alpha=1.6, levels=5, verbose=True):
    fixed_np = to_numpy(fixed)
    moving_np = to_numpy(moving)

    origin_type = moving_np.dtype

    shape = moving_np.shape

    fixed_np = fixed_np.flatten().astype(np.float32)
    moving_np = moving_np.flatten().astype(np.float32)
    moved_np = np.zeros(moving_np.shape).flatten().astype(np.float32)

    deeds_cpp(fixed_np, moving_np, moved_np, shape, alpha, levels, verbose)

    moved_np = np.reshape(moved_np, shape).astype(origin_type)

    moved = to_sitk(moved_np, ref_img=fixed)

    return moved


def to_numpy(img):
    result = sitk.GetArrayFromImage(img)

    return result


def to_sitk(img, ref_img=None):
    img = sitk.GetImageFromArray(img)

    if ref_img:
        img.CopyInformation(ref_img)

    return img
