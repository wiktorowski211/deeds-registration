import SimpleITK as sitk
import numpy as np
cimport numpy as np

cdef extern from "libs/deedsBCV0.h":
    int deeds(float *im1, float *im1b, float *warped1, int m, int n, int o, float alpha, int levels)


def deeds_cpp(np.ndarray[np.float32_t, ndim=1] moving,
              np.ndarray[np.float32_t, ndim=1] fixed,
              np.ndarray[np.float32_t, ndim=1] moved,
              shape, alpha, level):
    return deeds(&fixed[0], &moving[0], &moved[0],
                  shape[2], shape[1], shape[0],
                  alpha, level)


def registration(moving, fixed, verbose=True, alpha=1.6, levels=5):
    moving_np = to_numpy(moving)
    fixed_np = to_numpy(fixed)

    shape = moving_np.shape

    moving_np = moving_np.flatten().astype(np.float32)
    fixed_np = fixed_np.flatten().astype(np.float32)
    moved_np = np.zeros(moving_np.shape).flatten().astype(np.float32)

    deeds_cpp(moving_np, fixed_np, moved_np, shape, alpha, levels)

    moved_np = np.reshape(moved_np, shape)

    moved = to_sitk(moved_np, ref_img=moving)

    return moved


def to_numpy(img):
    result = sitk.GetArrayFromImage(img)

    return result


def to_sitk(img, ref_img=None):
    img = sitk.GetImageFromArray(img)

    if ref_img:
        img.CopyInformation(ref_img)

    return img
