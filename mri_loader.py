import os

import pandas as pd
import numpy as np

from stats import *

import nibabel
from nilearn import image
from glob import glob

from nilearn.image import concat_imgs

confound_columns = \
    ['a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03',
     'a_comp_cor_04', 'a_comp_cor_05', 'cosine00', 'cosine01', 'cosine02',
     'cosine03', 'cosine04', 'cosine05', 'trans_x', 'trans_y', 'trans_z',
     'rot_x', 'rot_y', 'rot_z']


reduced_columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']


class Subject:

    def __init__(self, subject_id, run_ids,
                 folder=None,
                 use_cache=True,
                 confound_mode='full',
                 volumes_offset=0):

        self.subject_id = subject_id
        self.run_ids = run_ids
        self.folder = folder
        self._dataset = [MRI(subject_id, run_id,
                             use_cache=use_cache,
                             folder=folder,
                             confound_mode=confound_mode,
                             volumes_offset=volumes_offset) for run_id in run_ids]

    def load(self):
        [mri.load() for mri in self._dataset]

    def compute_inflexions(self, low_percentile=25, high_percentile=75):
        all_labels = pd.concat([ds.labels for ds in self._dataset], ignore_index=True)

        cur_mean, cur_std = compute_morph_scores(all_labels)

        _, fitted_curve = fit_sigmoid(cur_mean)
        curve = np.array(cur_mean["response"])

        low_quartile = np.percentile(fitted_curve, low_percentile)
        low_inflexion = find_inflexion(curve, low_quartile)

        high_quartile = np.percentile(fitted_curve, high_percentile)
        high_inflexion = find_inflexion(curve, high_quartile)

        return low_inflexion, high_inflexion

    @property
    def sample_mask(self):
        masks = []
        last_index = 0

        for ds in self._dataset:
            masks.append(ds.sample_mask + last_index)
            last_index += ds.sample_mask[-1]

        return np.concatenate(masks)

    @property
    def brain_mask(self):
        return self._dataset[0].brain_mask

    @property
    def background(self):
        return self._dataset[0].background

    @property
    def repetition_time(self):
        return self._dataset[0]._t_r

    def get_durations(self, fill_nan=2.0, exclude_couples=False):
        durations = []

        for run in self._dataset:
            run_labels = run.labels

            if exclude_couples:
                run_labels = run_labels[~run_labels['couple'].isin(self.get_excluded_couples())]

            d = np.nan_to_num(run_labels["response time"].values, nan=fill_nan)
            durations.append(d)

        return np.concatenate(durations)

    def get_excluded_couples(self):
        with open(f"{self.folder}/labels/exclusion/couples_{self.subject_id}.csv") as f:
            couples_data = f.readlines()
        return [int(float(line.strip())) for line in couples_data if line.strip()]

    def get_data(self,
                 labels_col="morph level",
                 morph_response=False,
                 shift_onset_response=False,
                 exclude_couples=False,
                 scale=1):
        images = []
        times = []
        labels = []

        last_timestamp = 0

        for run in self._dataset:
            images.append(run.data)

            run_labels = run.labels

            if exclude_couples:
                run_labels = run_labels[~run_labels['couple'].isin(self.get_excluded_couples())]

            if shift_onset_response:
                r_time = run_labels["run time"].values
                resp_time = np.nan_to_num(run_labels["response time"].values)

                times.append(r_time + resp_time + last_timestamp)
            else:
                times.append(run_labels["run time"] + last_timestamp)

            last_timestamp += (run.data.shape[3] * self.repetition_time) * 1000

            if not morph_response:
                labels.append(run_labels[labels_col].values)
            else:
                original_labels = run_labels[labels_col].values
                responses = run_labels["response"].values

                new_labels = [f'{label}_{resp}' for label, resp in zip(original_labels, responses)]
                labels.append(new_labels)

        images = concat_imgs(images)
        images = image.math_img(f"img * {scale}", img=images)

        # convert ms to seconds
        times = np.concatenate(times) / 1000
        labels = np.concatenate(labels)

        return images, times, labels


class MRI:

    def __init__(self, subject_id, run_id,
                 folder=None,
                 sub_folder=True,
                 use_cache=True,
                 confound_mode='full',
                 volumes_offset=0):

        if folder is None:
            folder = "."

        self._sub_folder = ''
        self.volumes_offset = volumes_offset

        confound_cols = {"full": confound_columns, "reduced": reduced_columns}
        self.confounds_columns = confound_cols[confound_mode]
        self.confounds_mode = confound_mode

        if sub_folder:
            self._sub_folder = '/Familiarity'

        self.folder = folder
        self.subject_id = subject_id
        self.run_id = run_id
        self.mri_file_prefix = self._get_prefix(subject_id, run_id)

        self._use_cache = use_cache

        self.mri_timestamps = None
        self._brain_mask = None
        self._raw_labels = None
        self._mri_labels = None
        self._bg_mask = None
        self._cleaned = None
        self._t_r = None

        self._standardize = "zscore_sample"
        self._detrend = True
        self._low_pass = 0.08
        self._high_pass = 0.009

    def load(self):
        assert self.data is not None
        assert self.labels is not None
        assert self.background is not None
        assert self.brain_mask is not None

    def _get_prefix(self, subject_id, run_id):
        return f"{self.folder}{self._sub_folder}/sub-{subject_id:02}/func/sub-{subject_id:02}_task-morph_run-{run_id}"

    @property
    def labels(self):
        if self._raw_labels is None:
            labels = pd.read_csv(f"{self.folder}/labels/labels_{self.subject_id}.csv")
            labels = labels[labels["run"] == self.run_id]

            self._raw_labels = labels

        return self._raw_labels

    @property
    # @deprecated("Interpolation of labels with GLM doesn't work")
    def mri_labels(self):
        if self._mri_labels is None:
            record_length = self.data.shape[3]
            time_interval = self._t_r

            self.mri_timestamps = np.arange(time_interval, (record_length + 1) * time_interval, time_interval)

            times_df = pd.DataFrame({'mri_timestamps': np.astype(self.mri_timestamps * 1000, np.int64)})

            labels = self.labels
            self._mri_labels = pd.merge_asof(times_df, labels, left_on='mri_timestamps', right_on='run time', direction='backward')

        return self._mri_labels

    @property
    def confounds(self):
        return pd.read_csv(self._get_file('desc-confounds_timeseries.tsv'), delimiter='\t').iloc[self.volumes_offset:].reset_index(drop=True)

    @property
    def sample_mask(self):
        cf_df = self.confounds
        outlier_cols = [col for col in cf_df.columns if col.startswith('motion_outlier')]
        outlier_mask = cf_df[outlier_cols].any(axis=1)
        return np.where(~outlier_mask)[0]

    @property
    def brain_mask(self):
        if self._brain_mask is None:
            self._brain_mask = nibabel.load(self._get_file('brain_mask.nii.gz'))

        return self._brain_mask

    @property
    def background(self):
        if self._bg_mask is None:
            path = f"{self.folder}{self._sub_folder}/sub-{self.subject_id:02}/anat/sub-{self.subject_id:02}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
            self._bg_mask = nibabel.load(path)

        return self._bg_mask

    @property
    def preprocessed(self):
        data = nibabel.load(self._get_file('preproc_bold.nii.gz'))
        self._t_r = data.header.get_zooms()[3]

        return data

    def _get_file(self, pattern):
        files = glob(f"{self.mri_file_prefix}*{pattern}")
        if not files or len(files) > 1:
            raise ValueError(f"File not found for {pattern=} in {self.mri_file_prefix=}")
        return files[0]

    @property
    def data(self):
        if self._cleaned is None and os.path.exists(self.cache_path) and self._use_cache:
            self._cleaned = nibabel.load(self.cache_path)
            self._t_r = self._cleaned.header.get_zooms()[3]

            data = self._cleaned.get_fdata()
            data_trimmed = data[:, :, :, self.volumes_offset:]
            new_img = nibabel.Nifti1Image(data_trimmed, self._cleaned.affine, self._cleaned.header)
            self._cleaned = new_img

        if self._cleaned is None:
            taken = set(self.confounds_columns)
            present = set(self.confounds.columns)
            index = list(taken & present)
            missing = list(taken - present)
            if missing:
                print(f"Confound columns are missing: {', '.join(missing)}")

            confound_matrix = self.confounds[index].values
            data = self.preprocessed

            self._cleaned = image.clean_img(data,
                                            confounds=confound_matrix,
                                            standardize=self._standardize,
                                            detrend=self._detrend,
                                            low_pass=self._low_pass,
                                            high_pass=self._high_pass,
                                            t_r=self._t_r)

        return self._cleaned

    @property
    def cache_path(self):
        return f"{self.folder}/cache/{self.confounds_mode}_confounds/sub-{self.subject_id}-run-{self.run_id}.nii.gz"

    def cache(self, override_cache=False):
        os.makedirs(f"{self.folder}/cache", exist_ok=True)
        if not os.path.exists(self.cache_path) or override_cache:
            nibabel.save(self.data, self.cache_path)

