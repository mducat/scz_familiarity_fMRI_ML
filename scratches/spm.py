
import sys

sys.path.append("..")

from mri_loader import Subject, MRI

imgs, times, labels = Subject(6, [1], folder="..").get_data(labels_col="response")

print(imgs.shape)
print(times, labels)

import numpy as np

r = times[np.where(labels == 1)]

print(r.tolist())

c = MRI(6, 1, folder="..").confounds[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']]

print(c)
np.savetxt(
    "D:\\SPM\\confounds.txt",
    c.values,
    delimiter="\t",
    fmt="%.6f"
)