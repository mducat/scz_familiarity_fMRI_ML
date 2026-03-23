
import sys
import pandas as pd
from collections import Counter

from mri_loader import MRI

sys.path.append("..")


def get_indices(lines, reg):
    for i, l in enumerate(lines):
        if l.find(reg) != -1:
            yield i


def check(sub_id):
    lines = open(f"../labels/raw/labels_{sub_id}.csv").readlines()

    x = list(get_indices(lines, "Debut_run"))
    print(x)

    times = []
    deltas = []
    deltas_2 = []
    wtf = []
    category = []

    trig_delta = []

    for line in lines[x[0]:x[1]]:
        items = line.split(",")
        times.append(float(items[0]))
        deltas.append(float(items[1].replace("[", "").replace("]", "")))
        wtf.append(items[2])


        cat = items[3].split(" ")[0]
        category.append(cat)

        try:
            val = float(items[3].split(" ")[-1])
            deltas_2.append(val)

            if cat == "trigger":
                trig_delta.append(val)
        except:
            ...

    counts = Counter(category)

    counts = {}
    for item in category:
        counts[item] = counts.get(item, 0) + 1

    from pprint import pprint
    pprint(counts)

    # print(235920 - 233520)

    trig_values = set()

    for prev_trig, next_trig in zip(trig_delta[:-1], trig_delta[1:]):
        trig_values.add(next_trig - prev_trig)

    print(trig_values)

    mri = MRI(sub_id, 1, folder="..")
    run_len = mri.data.shape[3] * mri._t_r
    print(mri.data.shape, mri._t_r)

    print(run_len)

    print(509514 - 34325)
    print(35172 - 34325)


check(1)
