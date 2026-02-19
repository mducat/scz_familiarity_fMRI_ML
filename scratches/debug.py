
import numpy as np
import pandas as pd


lines = open("../labels/labels_1.csv").readlines()


all_resp = []
all_time = []
for l in lines[1:]:
    dat = l.split(",")
    resp = dat[6]
    time = dat[7].strip()
    morph = dat[4]

    if morph == "5":
        all_resp.append(resp)
        all_time.append(time)


print(all_time)
all_resp = [int(v) for v in all_resp]
all_time = [int(v) for v in all_time if v]
# print(sum(all_resp) / len(all_resp))
print(np.std(all_time, ddof=1) / np.sqrt(len(all_time)))

x= pd.read_csv("../labels/labels_1.csv")
print(x.groupby(["morph level"]).std()[["response time"]].iloc[0, 0])




from glob import glob

x = glob("../Familiarity/sub-11/func/sub-11_task-morph_run-1*")
x = [v.split("\\")[-1] for v in x]

print(x)


import nibabel

#x = nibabel.load("../Familiarity/sub-13/func/sub-13_task-morph_run-5_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")

#print(x.get_fdata())

import sys

sys.path.append("..")

import mri_loader

m = mri_loader.MRI(11, 2, folder="..")
#m.load()

print(m._get_file("MNI152NLin2009cAsym_boldref.nii.gz"))
print(m.background)


