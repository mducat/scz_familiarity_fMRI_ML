import os

import pandas as pd
import numpy as np

import nibabel
from nilearn import image
from glob import glob

from typing_extensions import deprecated

confound_columns = \
    ['a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03',
     'a_comp_cor_04', 'a_comp_cor_05', 'cosine00', 'cosine01', 'cosine02',
     'cosine03', 'cosine04', 'cosine05', 'trans_x', 'trans_y', 'trans_z',
     'rot_x', 'rot_y', 'rot_z']



class MRI:

    def __init__(self, subject_id, run_id, folder=None):
        if folder is None:
            folder = "."

        self.folder = folder
        self.subject_id = subject_id
        self.run_id = run_id
        self.mri_file_prefix = self._get_prefix(subject_id, run_id)

        self.mri_timestamps = None
        self._brain_mask = None
        self._raw_labels = None
        self._mri_labels = None
        self._bg_mask = None
        self._cleaned = None
        self._t_r = None

        self._low_pass = 0.08
        self._high_pass = 0.009

    def load(self):
        assert self.data is not None
        assert self.labels is not None
        assert self.background is not None
        assert self.brain_mask is not None

    def _get_prefix(self, subject_id, run_id):
        return f"{self.folder}/Familiarity/sub-{subject_id}/func/sub-{subject_id}_task-morph_run-{run_id}"

    @property
    def labels(self):
        if self._raw_labels is None:
            labels = pd.read_csv(f"{self.folder}/labels/labels_{self.subject_id}.csv")
            labels = labels[labels["run"] == self.run_id]

            self._raw_labels = labels

        return self._raw_labels

    @property
    @deprecated("Interpolation of labels with GLM doesn't work")
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
        return pd.read_csv(self._get_file('desc-confounds_timeseries.tsv'), delimiter='\t')

    @property
    def brain_mask(self):
        if self._brain_mask is None:
            self._brain_mask = nibabel.load(self._get_file('brain_mask.nii.gz'))

        return self._brain_mask

    @property
    def background(self):
        if self._bg_mask is None:
            path = f"{self.folder}/Familiarity/sub-{self.subject_id}/anat/sub-{self.subject_id}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
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
            return None
        return files[0]

    @property
    def data(self):
        if self._cleaned is None and os.path.exists(self.cache_path):
            self._cleaned = nibabel.load(self.cache_path)
            self._t_r = self._cleaned.header.get_zooms()[3]

        if self._cleaned is None:
            taken = set(confound_columns)
            present = set(self.confounds.columns)
            index = list(taken & present)

            confound_matrix = self.confounds[index].values
            data = self.preprocessed

            self._cleaned = image.clean_img(data,
                                            confounds=confound_matrix,
                                            standardize='zscore_sample',
                                            detrend=True,
                                            low_pass=self._low_pass,
                                            high_pass=self._high_pass,
                                            t_r=self._t_r)

        return self._cleaned

    @property
    def cache_path(self):
        return f"{self.folder}/cache/sub-{self.subject_id}-run-{self.run_id}.nii.gz"

    def cache(self):
        os.makedirs(f"{self.folder}/cache", exist_ok=True)
        nibabel.save(self.data, self.cache_path)
