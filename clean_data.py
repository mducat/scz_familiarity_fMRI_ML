
from mri_loader import MRI


subs = set(range(22,33))
# subs ^= {}


for sub in subs:  # [11,12,13]:
    for run in [1,2,3,4]:  # [1,2,3,4,5]:

        print(f'clean {sub} run {run}')

        try:
            data = MRI(sub, run)
            data.load()
            data.cache()
        except Exception as e:
            print(e)
            # raise


