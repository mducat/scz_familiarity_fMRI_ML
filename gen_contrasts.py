import os
from dataclasses import dataclass

from mri_loader import Subject
from nilearn.glm.first_level import FirstLevelModel

from nilearn.reporting import get_clusters_table
from nilearn.glm import threshold_stats_img

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from lib.mni_to_atlas import AtlasBrowser

from stats import *
import nibabel


@dataclass
class Config:
    # SUBJECTS = 'SCZ'
    SUBJECTS = 'CONTROL'
    EXCLUDE_WITH_SIGMOID = True

    VOLUMES_OFFSET = 0
    CONFOUND_MODE = 'full'
    USE_SAMPLE_MASKS = True
    SMOOTHING_FWHM = 5
    DURATION = 2.5

    PREDICTORS = "morph_with_response"
    # PREDICTORS = "morph"
    # PREDICTORS = "response"

    CORRECTIONS = [[0.05, "bonferroni", 2], [0.001, "fdr", 5]]

    SAVE_CONTRASTS = True


run_ids = [1, 2, 3, 4]



def exclude_with_sigmoid(subject_ids):
    exclude_inflexion = set()

    for subject in subject_ids:
        try:
            dataset = Subject(subject, run_ids)

            low_inflexion, high_inflexion = dataset.compute_inflexions()

            if low_inflexion < 0.1 or high_inflexion > 0.9 or high_inflexion < 0.5 or low_inflexion > 0.5:
                exclude_inflexion.add(subject)
        except Exception as e:
            print(e)
            exclude_inflexion.add(subject)
            continue

    return exclude_inflexion


def gen_contrast_list():
    contrast_list = [{"+": ["high"], "-": ["low"]},  # high > low
                     {"+": ["undecided"], "-": ["high", "low"]}, ]  # undecided > high + low

    scale = [["25", "35"], ["35", "45"], ["45", "55"], ["55", "65"], ["65", "75"], ["75", "85"], ["85", "95"]]
    to_subtract = {"-": ["5", "15"]}

    for values in scale:
        contrast_list.append({
            "+": values,
            **to_subtract
        })

    contrast_list += [{"+": ["button"], "-": ["unpressed"]}]

    return contrast_list


def GLM_contrast_map(cfg, global_z_map, subject_id, labels_col, morph_response):
    dataset = Subject(subject_id, run_ids, confound_mode=cfg.CONFOUND_MODE, volumes_offset=cfg.VOLUMES_OFFSET)
    dataset.load()

    images, times, labels = dataset.get_data(labels_col=labels_col, morph_response=morph_response)
    low_inflexion, high_inflexion = dataset.compute_inflexions()

    sample_mask = dataset.sample_mask

    print(f"{subject_id=} {low_inflexion=}, {high_inflexion=}")

    events = pd.DataFrame(
        {'onset': times,
         'trial_type': labels,
         'duration': cfg.DURATION}
    )

    repetition_time = dataset.repetition_time
    fmri_glm = FirstLevelModel(t_r=repetition_time,
                               drift_model='polynomial',
                               drift_order=3,
                               hrf_model='spm',
                               mask_img=dataset.brain_mask,
                               smoothing_fwhm=cfg.SMOOTHING_FWHM,
                               n_jobs=-1)

    if cfg.USE_SAMPLE_MASKS:
        fmri_glm = fmri_glm.fit(images, events, sample_masks=sample_mask)
    else:
        fmri_glm = fmri_glm.fit(images, events)

    design_matrix = fmri_glm.design_matrices_[0]

    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = {
        str(column): contrast_matrix[i]
        for i, column in enumerate(design_matrix.columns)
    }

    if labels_col == "morph level":
        parse_contrast(contrasts, low_inflexion, high_inflexion)

    for contrast in gen_contrast_list():

        glm_contrast_vector = np.sum(contrasts[column] for column in contrast["+"])
        if '-' in contrast:
            glm_contrast_vector -= np.sum(contrasts[column] for column in contrast["-"])

        z_score = fmri_glm.compute_contrast(glm_contrast_vector, output_type="z_score")
        name = contrast_name(contrast)

        global_z_map[name].append(z_score)


atlas = AtlasBrowser("AAL3")

def get_regions(global_z_map, correction):
    mni_regions = {}
    alpha, method, cluster_size = correction

    for c_name, images in global_z_map.items():
        mni_regions[c_name] = []

        for z_score in images:

            clean, threshold = threshold_stats_img(z_score,
                                                   alpha=alpha,
                                                   height_control=method,
                                                   cluster_threshold=cluster_size)

            table = get_clusters_table(clean, stat_threshold=threshold, cluster_threshold=cluster_size)

            pos = [np.array([x, y, z]) for (x, y, z) in zip(table['X'], table['Z'], table['Y'])]

            for p in pos:
                try:
                    projected_coords = atlas.project_to_nearest(p)
                    projected_regions = atlas.find_regions(projected_coords)

                    mni_regions[c_name].append(*projected_regions)
                except Exception as e:
                    print(e)

    return mni_regions


def plot_regions(cfg, mni_regions, correction, subject_ids):
    mni_regions_concat = {}

    for k, v in mni_regions.items():
        reg_list = np.array(v, dtype=object)

        mni_regions_concat[k] = reg_list

    mni_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in mni_regions_concat.items()]))
    mni_summary = pd.DataFrame({
        col.replace(' ', ''): mni_df[col].value_counts()
        for col in mni_df.columns
    })

    mni_summary = mni_summary.fillna(0)

    fig, ax = plt.subplots(figsize=(18, 30))
    im = ax.imshow(mni_summary.values, cmap='Wistia', aspect='auto')

    ax.set_xticks(range(len(mni_summary.columns)))
    ax.set_yticks(range(len(mni_summary)))
    ax.set_xticklabels(mni_summary.columns, fontsize=10)
    ax.set_yticklabels(mni_summary.index, fontsize=14)

    ax.xaxis.tick_top()

    for i in range(len(mni_summary.index)):
        for j in range(len(mni_summary.columns)):
            ax.text(j, i, f'{mni_summary.values[i, j]:.0f}',
                    ha='center', va='center', fontsize=10)

    ax.grid(False)
    ax.tick_params(which='both', length=0)

    subject_parameters = f"{cfg.SUBJECTS=}, {cfg.EXCLUDE_WITH_SIGMOID=}, {len(subject_ids)=}"
    glm_parameters = f"{cfg.VOLUMES_OFFSET=}, {cfg.CONFOUND_MODE=}, {cfg.USE_SAMPLE_MASKS=}, {cfg.SMOOTHING_FWHM=}, {cfg.DURATION=}"
    contrast_parameters = f"{cfg.PREDICTORS=}, {correction=}"

    text = f"{subject_parameters}\n{glm_parameters}\n{contrast_parameters}"

    fig.subplots_adjust(bottom=0.05)
    fig.text(0.5, 0.02, text, ha="center", fontsize=12, color="gray", style="italic")

    plt.colorbar(im)


def path(cfg):
    mask_name = 'masked' if cfg.USE_SAMPLE_MASKS else 'unmasked'
    filtered = 'filtered' if cfg.EXCLUDE_WITH_SIGMOID else 'unfiltered'

    dur = f"d{str(cfg.DURATION).replace('.', '-')}"

    return f"{cfg.SUBJECTS}_{cfg.CONFOUND_MODE}_{cfg.VOLUMES_OFFSET}_{mask_name}_{filtered}_{cfg.SMOOTHING_FWHM}_{dur}"


def run(cfg):

    subjects_ids_per_type = {
        "SCZ": set(range(27, 34)),
        "CONTROL": set(range(1, 27))
    }

    subject_ids = subjects_ids_per_type[cfg.SUBJECTS]

    if cfg.EXCLUDE_WITH_SIGMOID:
        subject_ids -= exclude_with_sigmoid(subject_ids)

    skipped = []

    labels_col = "morph level"
    morph_response = False

    if cfg.PREDICTORS == "morph_with_response":
        morph_response = True
    elif cfg.PREDICTORS == "response":
        labels_col = "response"

    global_z_map = defaultdict(list)

    for subject in subject_ids:

        try:
            GLM_contrast_map(cfg, global_z_map, subject, labels_col, morph_response)
        except Exception as e:
            print("Skipping subject ", subject)
            print(e)
            skipped.append(subject)
        continue

    filepath = path(cfg)

    os.makedirs(f"brute_force/{filepath}/regions/", exist_ok=True)
    os.makedirs(f"brute_force/{filepath}/contrasts/", exist_ok=True)

    for correction in cfg.CORRECTIONS:

        mni_regions = get_regions(global_z_map, correction)
        plot_regions(cfg, mni_regions, correction, subject_ids)

        cor_name = '_'.join(str(c) for c in correction).replace('0.', "p")

        plt.savefig(f"brute_force/{filepath}/regions/{cor_name}.png")

    z_map_subjects = subject_ids - set(skipped)
    for c_name, images in global_z_map.items():
        for z_score, subject in zip(images, z_map_subjects):

            fname = c_name.replace(' ', '_').replace('>', 'over')

            nibabel.save(z_score, f"brute_force/{filepath}/contrasts/sub-{subject}-{fname}.nii.gz")


if __name__ == '__main__':

    from itertools import product
    import pickle

    cf_mode = ['full', 'reduced']
    masks = [True, False]
    smoothing = [3, 5, 7]
    duration = [2.5, 5, 7.5]

    todo = set(product(cf_mode, masks, smoothing, duration))

    with open("cache.pkl", "rb") as f:
        done = pickle.load(f)

    todo -= done

    cfg = Config()

    for combination in todo:

        cfg.CONFOUND_MODE = combination[0]
        cfg.USE_SAMPLE_MASKS = combination[1]
        cfg.SMOOTHING_FWHM = combination[2]
        cfg.DURATION = combination[3]

        run(cfg)

        done.add(combination)

        with open("cache.pkl", "wb") as f:
            pickle.dump(done, f)

