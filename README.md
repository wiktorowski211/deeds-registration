# DEnse Displacement Sampling - deformable image registration

This package provides Python wrapper around [DEEDS](https://github.com/mattiaspaul/deedsBCV), an efficient version for 3D discrete deformable image registration which is reaching the highest accuracy in several benchmarks [[1]](https://pubmed.ncbi.nlm.nih.gov/27254856/)[[2]](https://arxiv.org/abs/2109.11572) and serves as a good baseline for new solutions.

## Referencing and citing
If you use this implementation or parts of it please cite:
 
>"MRF-Based Deformable Registration and Ventilation Estimation of Lung CT."
 by Mattias P. Heinrich, M. Jenkinson, M. Brady and J.A. Schnabel
 IEEE Transactions on Medical Imaging 2013, Volume 32, Issue 7, July 2013, Pages 1239-1248
 http://dx.doi.org/10.1109/TMI.2013.2246577
 
 and
 
>"Multi-modal Multi-Atlas Segmentation using Discrete Optimisation and Self-Similarities"
 by Mattias P. Heinrich, Oskar Maier and Heinz Handels
 VISCERAL Challenge@ ISBI, Pages 27-30 2015
 http://ceur-ws.org/Vol-1390/visceralISBI15-4.pdf
 
## Installation
Current version requires an Unix-based system.
```
pip install git+https://github.com/wiktorowski211/deeds-registration
```

## Usage
```
from deeds import registration
import SimpleITK as sitk

fixed = sitk.ReadImage(PATH)
moving = sitk.ReadImage(PATH)

moved = registration(fixed, moving)
```

## Limitations
Input images must:
- have the same dimensions,
- be a SimpleITK image object.
