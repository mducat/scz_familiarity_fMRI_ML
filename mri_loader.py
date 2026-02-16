
import os

import nibabel
from nilearn import datasets, image
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker, NiftiMapsMasker

confound_columns = \
    ['a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03',
     'a_comp_cor_04', 'a_comp_cor_05', 'cosine00', 'cosine01', 'cosine02',
     'cosine03', 'cosine04', 'cosine05', 'trans_x', 'trans_y', 'trans_z',
     'rot_x', 'rot_y', 'rot_z']


class MRI:

    def __init__(self, filename, result_path):
        self.data_path = os.path.dirname(filename)
        self.filename = filename
        self.result_path = result_path
        assert os.path.exists(self.filename)
        self._nifti_filename = None
        self._data = None
        self._brain_mask = None
        self._mask_indices = None
        self._low_pass = 0.08
        self._high_pass = 0.009

        self._t_r = self.data.header.get_zooms()[3]

        if self.is_preprocessed:
            self._masker = NiftiMasker(mask_img=self.brain_mask, standardize=True, mask_strategy='epi')

            self.dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            self.atlas_filename = self.dataset.filename
            self.labels = self.dataset.labels[1:]

            self._parcelizer = NiftiLabelsMasker(labels_img=self.atlas_filename, standardize=True,
                                                 memory='nilearn_cache', verbose=5, mask_img=self.brain_mask)

    @property
    def confounds(self):
        import pandas as pd
        if self.is_preprocessed:
            return pd.read_csv(self._get_file('desc-confounds_timeseries.tsv'), delimiter='\t')
        else:
            return None

    @property
    def data(self):
        if self._data is None:
            self._data = nibabel.load(self.filename)

        return self._data

    @property
    def brain_mask(self):
        if not self.is_preprocessed:
            return None

        if self._brain_mask is None:
            self._brain_mask = nibabel.load(self._get_file('brain_mask.nii.gz'))

        return self._brain_mask

    def _get_file(self, pattern):
        files = os.listdir(self.result_path)

        for file in files:
            if file.find(pattern) > -1:
                return os.path.join(self.result_path, file)

        return None

    @property
    def is_preprocessed(self):
        files = os.listdir(self.result_path)
        for file in files:
            if file.find('MNI152') > -1:
                return True
        print("Files should be preprocessed via fmriprep first!")
        return False

    @property
    def cleaned(self):
        if not self.is_preprocessed:
            return None

        confound_matrix = self.confounds[self.confound_columns].values
        return image.clean_img(self.preprocessed, confounds=confound_matrix,
            detrend=True, low_pass=self._low_pass, high_pass=self._high_pass, t_r=self._t_r)
