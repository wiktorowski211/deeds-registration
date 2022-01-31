import unittest

from ..registration import registration
from ..utils import *


class TestStringMethods(unittest.TestCase):

    def test_deeds_registration(self):
        fixed = load_nifty('samples/fixed.nii.gz')
        moving = load_nifty('samples/moving.nii.gz')

        moved = registration(fixed, moving)

        dice_before = compute_dice_score(fixed, moving)
        dice_after = compute_dice_score(fixed, moved)

        self.assertGreater(dice_after, dice_before)
