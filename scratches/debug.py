
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
