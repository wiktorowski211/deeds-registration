import unittest
import SimpleITK as sitk

from ..registration import registration


class TestStringMethods(unittest.TestCase):

    def test_deeds_registration(self):
        fixed = load_nifty('samples/fixed.nii.gz')
        moving = load_nifty('samples/moving.nii.gz')

        moved = registration(fixed, moving)

        dice_before = compute_dice_score(fixed, moving)
        dice_after = compute_dice_score(fixed, moved)

        self.assertGreater(dice_after, dice_before)


def load_nifty(path):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(path)
    return reader.Execute()


def compute_dice_score(img1, img2, threshold=127):
    img2.SetOrigin(img1.GetOrigin())

    img1 = sitk.BinaryThreshold(img1, lowerThreshold=threshold)
    img2 = sitk.BinaryThreshold(img2, lowerThreshold=threshold)

    metrics = sitk.LabelOverlapMeasuresImageFilter()
    metrics.Execute(img1, img2)

    return metrics.GetDiceCoefficient()
