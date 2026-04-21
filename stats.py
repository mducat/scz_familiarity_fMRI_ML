from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

from nilearn.maskers import NiftiMasker

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


def plot_r2(r2_values):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"R² Values Distribution (n={len(r2_values)})", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(r2_values, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
    ax1.axvline(np.mean(r2_values), color="red", linestyle="--", label=f"Mean:   {np.mean(r2_values):.3f}")
    ax1.axvline(np.median(r2_values), color="orange", linestyle="--", label=f"Median: {np.median(r2_values):.3f}")
    ax1.set_title("Histogram")
    ax1.set_xlabel("R²")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    kde = gaussian_kde(r2_values)
    x = np.linspace(r2_values.min(), r2_values.max(), 500)
    ax2.plot(x, kde(x), color="steelblue", linewidth=2)
    ax2.fill_between(x, kde(x), alpha=0.3, color="steelblue")
    ax2.axvline(np.mean(r2_values), color="red", linestyle="--", label=f"Mean:   {np.mean(r2_values):.3f}")
    ax2.axvline(np.median(r2_values), color="orange", linestyle="--", label=f"Median: {np.median(r2_values):.3f}")
    ax2.set_title("Kernel Density Estimate")
    ax2.set_xlabel("R²")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.boxplot(r2_values, vert=False, patch_artist=True,
                boxprops=dict(facecolor="steelblue", alpha=0.5),
                medianprops=dict(color="orange", linewidth=2))
    sample = np.random.choice(r2_values, size=min(500, len(r2_values)), replace=False)
    ax3.scatter(sample, np.ones(len(sample)),
                alpha=0.15, color="steelblue", s=10, zorder=3)
    ax3.set_title("Box + Strip Plot")
    ax3.set_xlabel("R²")
    ax3.set_yticks([])

    ax4 = fig.add_subplot(gs[1, 1])
    sorted_vals = np.sort(r2_values)
    ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax4.plot(sorted_vals, ecdf, color="steelblue", linewidth=2)
    ax4.axhline(0.5, color="orange", linestyle="--", label="50th percentile")
    ax4.axhline(0.25, color="gray", linestyle=":", label="25th percentile")
    ax4.axhline(0.75, color="gray", linestyle=":", label="75th percentile")
    ax4.set_title("Empirical CDF (ECDF)")
    ax4.set_xlabel("R²")
    ax4.set_ylabel("Cumulative Probability")
    ax4.legend(fontsize=9)

    stats_text = (
        f"Min:    {r2_values.min():.3f}\n"
        f"Max:    {r2_values.max():.3f}\n"
        f"Mean:   {np.mean(r2_values):.3f}\n"
        f"Median: {np.median(r2_values):.3f}\n"
        f"Std:    {np.std(r2_values):.3f}\n"
        f"<0:     {(r2_values < 0).sum()} values\n"  # worse than baseline
        f">0:     {(r2_values > 0).sum()} values"  # sanity check
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.show()


def carpet_plot(fmri_img, mask_img, t_r=2.4, standardize=True,
                title="Carpet Plot", figsize=(14, 8)):

    masker = NiftiMasker(
        mask_img=mask_img,
        standardize=standardize,
        t_r=t_r
    )
    voxels = masker.fit_transform(fmri_img)  # shape: (timepoints, voxels)
    # print(f"Data shape → timepoints: {voxels.shape[0]}, voxels: {voxels.shape[1]}")

    global_signal = voxels.mean(axis=1)

    sort_idx = np.argsort(voxels.mean(axis=0))
    voxels_sorted = voxels[:, sort_idx].T  # shape: (voxels, timepoints)

    n_timepoints = voxels.shape[0]
    time_axis = np.arange(n_timepoints) * t_r

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        2, 1,
        height_ratios=[1, 5],  # global signal : carpet
        hspace=0.05
    )

    ax_gs = fig.add_subplot(gs[0])
    ax_gs.plot(time_axis, global_signal, color="black", linewidth=0.8)
    ax_gs.set_xlim(time_axis[0], time_axis[-1])
    ax_gs.set_ylabel("Global\nSignal", fontsize=9)
    ax_gs.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_gs.set_xticklabels([])
    ax_gs.spines[["top", "right"]].set_visible(False)
    ax_gs.set_title(title, fontsize=12, fontweight="bold")

    ax_cp = fig.add_subplot(gs[1])

    vmin = np.percentile(voxels_sorted, 2)
    vmax = np.percentile(voxels_sorted, 98)

    im = ax_cp.imshow(
        voxels_sorted,
        aspect="auto",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], 0, voxels_sorted.shape[0]]
    )

    ax_cp.set_xlabel("Time (s)", fontsize=10)
    ax_cp.set_ylabel("Voxels",   fontsize=10)
    ax_cp.spines[["top", "right"]].set_visible(False)

    cbar = fig.colorbar(im, ax=ax_cp, orientation="vertical",
                        fraction=0.02, pad=0.02)
    cbar.set_label("Signal\nIntensity", fontsize=8)

    plt.tight_layout()
    plt.show()

    return fig


def plot_timeseries_list(*timeseries, **kwargs):
    data = np.array([*timeseries])
    plot_timeseries(data, **kwargs)


def plot_timeseries(timeseries, labels=None, repetition_time=None, events=None):

    assert repetition_time is not None, "repetition_time is None"

    n_volumes = timeseries.shape[1]  # assuming shape (n_voxels, n_timepoints)
    time_volumes = np.arange(n_volumes) * repetition_time  # e.g. [0, 2, 4, 6, ...]
    time_seconds = np.arange(0, time_volumes[-1], 1)  # 1 second resolution

    timeseries = np.array([
        interp1d(time_volumes, timeserie, kind="linear")(time_seconds)
        for timeserie in timeseries
    ])

    fig, ax = plt.subplots(figsize=(15, 8))

    if labels is not None:
        assert len(labels) == timeseries.shape[0], f"len(labels) != len(timeseries), {len(labels)} != {timeseries.shape[0]}"

    for i, timeserie in enumerate(timeseries):
        kwargs = {}

        if labels:
            l = labels[i]
            kwargs["label"] = l

            if "predicted" in l:
                kwargs["linestyle"] = '--'

        ax.plot(timeserie, **kwargs)

    if events is not None:
        trial_types = events["trial_type"].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(trial_types)))
        color_map = dict(zip(trial_types, colors))

        for _, row in events.iterrows():
            xpos = row["onset"]

            ax.axvline(
                x=xpos,
                ymin=0.0,
                ymax=0.2,
                color=color_map[row["trial_type"]],
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
                label=row["trial_type"]
            )

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_handles, unique_labels = [], []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = True
            unique_handles.append(handle)
            unique_labels.append(label)

    ax.legend(unique_handles, unique_labels, loc="upper right", bbox_to_anchor=(1.15, 1))

    ax.set_xlabel(f"Time (Seconds)")
    ax.set_ylabel("BOLD Signal")
    ax.set_title("fMRI Timeseries with Event Onsets")
    plt.tight_layout()
    plt.show()


def generate_motor_mask(merged=True, override_roi=None, folder='.'):
    from nibabel.affines import apply_affine
    from mri_loader import MRI

    import nibabel as nib

    import lib.mni_to_atlas as mni_to_atlas

    mni_to_atlas._ATLASES_PATH = f"{folder}/lib"
    atlas = mni_to_atlas.AtlasBrowser("AAL3")

    ROI = ['Precentral_L', 'Precentral_R', 'Supp_Motor_Area_L', 'Supp_Motor_Area_R']
    if override_roi:
        ROI = override_roi

    sample = MRI(1, 1).data
    inv_affine = np.linalg.inv(sample.affine)

    masks = {}
    masks_image = {}
    lookup_regions = {v: k for k, v in atlas._region_names.items()}

    merged_mask = []

    for region in ROI:
        index = lookup_regions[region]

        atlas_mask = np.argwhere(atlas._image == index)
        mni_mask = atlas._convert_atlas_to_mni_space(atlas_mask)
        image_mask = apply_affine(inv_affine, mni_mask)
        cleaned_mask = np.unique(np.round(image_mask), axis=0).astype(np.int32)

        if merged:
            merged_mask.append(cleaned_mask)
        else:

            applied_mask = np.zeros(sample.shape[:3], dtype=np.uint8)
            applied_mask[cleaned_mask[:, 0], cleaned_mask[:, 1], cleaned_mask[:, 2]] = 1

            img = nib.Nifti1Image(applied_mask, affine=sample.affine)
            masks_image[region] = img
            masks[region] = NiftiMasker(mask_img=img)
            masks[region].fit()

    if merged:
        merged_mask = np.concatenate(merged_mask, axis=0)

        applied_mask = np.zeros(sample.shape[:3], dtype=np.uint8)
        applied_mask[merged_mask[:, 0], merged_mask[:, 1], merged_mask[:, 2]] = 1

        img = nib.Nifti1Image(applied_mask, affine=sample.affine)
        mask = NiftiMasker(mask_img=img)
        mask.fit()

        return mask, img

    return masks, masks_image


