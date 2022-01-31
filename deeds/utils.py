import SimpleITK as sitk


def load_nifty(path):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(path)
    return reader.Execute()


def save_nifty(img, path):
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(img)


def compute_dice_score(img1, img2, threshold=127):
    img2.SetOrigin(img1.GetOrigin())

    img1 = sitk.BinaryThreshold(img1, lowerThreshold=threshold)
    img2 = sitk.BinaryThreshold(img2, lowerThreshold=threshold)

    metrics = sitk.LabelOverlapMeasuresImageFilter()
    metrics.Execute(img1, img2)

    return metrics.GetDiceCoefficient()
