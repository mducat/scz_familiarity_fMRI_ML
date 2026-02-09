
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
    summed = grouped.sum()

    std /= np.sqrt(summed)

    return mean[columns], std[columns]


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y


def fit_sigmoid(mean):
    # returns the sigmoid parameters + standard curve
    # overflow in curve_fit
    import warnings
    warnings.filterwarnings("ignore")

    ydata = np.array(mean["response"])
    # fit scale is in 0.05-0.95, csv is in scale 5-95
    xdata = np.linspace(0.05, 0.95, 10)
    p0 = [-1, 0, 1, 1]  # pre-set parameters to get right curve orientation

    popt, _ = curve_fit(sigmoid, xdata, ydata, p0, full_output=False)

    return popt, sigmoid(xdata, *popt)


def find_inflexion(fitted_curve):
    xinterp = np.linspace(0.05, 0.95, 1000)
    xvals = np.linspace(0.05, 0.95, 10)
    interpolated = np.interp(xinterp, xvals, fitted_curve)

    point = xinterp[np.argwhere(np.diff(interpolated > 0.5))]

    if np.any(point):
        return point[0, 0]
    return None


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
        ax0.legend(loc="lower right")

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




