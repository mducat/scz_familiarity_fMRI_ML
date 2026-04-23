
from mri_loader import MRI


subs = set(range(1,34))

# subs = [1, 3, 4, 5, 20, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

still_fucked = [1, 3, 13, 22, 32, 33]


for sub in []:  # [11,12,13]:
    for run in [5]:  # [1,2,3,4,5]:

        print(f'clean {sub} run {run}')

        try:
            data = MRI(sub, run, confound_mode="reduced", use_cache=False)

            data._standardize = "psc"
            data._detrend = True

            data.load()
            data.cache(override_cache=True)
        except Exception as e:
            print(e)
            # raise


