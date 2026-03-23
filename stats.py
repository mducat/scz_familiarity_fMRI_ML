from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use('seaborn-v0_8')


def compute_morph_scores(df, columns=None):
    grouped = df.groupby(["morph level"])

    if not columns:  # default
        columns = ["response", "response time"]

    mean = grouped.mean()
    std = grouped.std()
    size = grouped.size()

    norm = np.array(np.sqrt(size))
    std = std.div(norm, axis=0)

    return mean[columns], std[columns]


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y


def fit_sigmoid(mean):
    # returns the sigmoid parameters + standard curve
    # overflow in curve_fit
    # import warnings
    # warnings.filterwarnings("ignore")

    ydata = np.array(mean["response"])
    # fit scale is in 0.05-0.95, csv is in scale 5-95
    xdata = np.linspace(0.05, 0.95, 10)
    p0 = [-1, 0, 1, 1]  # pre-set parameters to get right curve orientation

    popt, _ = curve_fit(sigmoid, xdata, ydata, p0, full_output=False)

    return popt, sigmoid(xdata, *popt)


def find_inflexion(fitted_curve, threshold=0.5):
    xinterp = np.linspace(0.05, 0.95, 1000)
    xvals = np.linspace(0.05, 0.95, 10)
    interpolated = np.interp(xinterp, xvals, fitted_curve)

    point = xinterp[np.argwhere(np.diff(interpolated > threshold))]

    if np.any(point):
        return point[0, 0]
    return None


def contrast_name(classes):
    positive = 'rest'
    negative = 'rest'
    if '+' in classes:
        positive = ' + '.join(classes["+"])
    if '-' in classes:
        negative = ' + '.join(classes["-"])
    return ' > '.join([positive, negative])


def parse_contrast(contrasts, low_inflexion, high_inflexion):
    low_contrast_columns = []
    high_contrast_columns = []
    undecided_contrast_columns = []
    button_contrast_columns = []
    unpressed_contrast_columns = []

    morph_columns = defaultdict(list)

    for key, column in contrasts.items():
        try:
            key_numeric = float(key.split("_")[0]) / 100
            key_parsed = str(int(key_numeric * 100))
            morph_columns[key_parsed].append(column)

            resp = int(key.split("_")[-1])
            if resp == 1:
                button_contrast_columns.append(column)
            else:
                unpressed_contrast_columns.append(column)

            if key_numeric < low_inflexion:
                low_contrast_columns.append(column)

            elif key_numeric > high_inflexion:
                high_contrast_columns.append(column)

            else:
                undecided_contrast_columns.append(column)

        except ValueError:
            continue

    contrasts["low"] = np.sum(low_contrast_columns, axis=0)
    contrasts["high"] = np.sum(high_contrast_columns, axis=0)
    contrasts["undecided"] = np.sum(undecided_contrast_columns, axis=0)
    contrasts["button"] = np.sum(button_contrast_columns, axis=0)
    contrasts["unpressed"] = np.sum(unpressed_contrast_columns, axis=0)

    for key, cols in morph_columns.items():
        contrasts[key] = np.sum(cols, axis=0)


def plot_behavioral_data(mean, std, sigmoid_curve=None, inflexion_point=None, title=None):

    morph = mean.index

    with plt.style.context('seaborn-v0_8'):
        fig, ax = plt.subplots(ncols=2, squeeze=False, figsize=(15, 5))

        if title:
            fig.suptitle(title, fontweight="bold")

        resp_mean = mean["response"]
        resp_std = std["response"]

        ax0 = ax[0, 0]
        ax0.plot(resp_mean, label="response")
        ax0.fill_between(morph, resp_mean - resp_std, resp_mean + resp_std, alpha=0.3)

        if sigmoid_curve is not None:
            ax0.plot(morph, sigmoid_curve, label="fitted sigmoid")

        if inflexion_point is not None:
            ax0.axvline(x=inflexion_point * 100, color="blue", linestyle="--", label=f"Inflexion threshold (>50%): {inflexion_point:.2f}")

        ax0.set_xticks(morph)
        ax0.set_xlabel("% morph (alpha)")
        ax0.set_ylabel("response")
        ax0.legend()

        time_mean = mean["response time"]
        time_std = std["response time"]

        ax1 = ax[0, 1]
        ax1.plot(time_mean, label="response time")
        ax1.fill_between(morph, time_mean - time_std, time_mean + time_std, alpha=0.3)

        ax1.set_xticks(morph)
        ax1.set_xlabel("% morph (alpha)")
        ax1.set_ylabel("response time (ms)")
        ax1.legend(loc="upper right")

    plt.show()




