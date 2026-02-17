
from mri_loader import MRI


for sub in [12,13]:  # [11,12,13]:
    for run in [1,2,3,4,5]:  # [1,2,3,4,5]:

        data = MRI(sub, run)
        data.load()
        data.cache()


