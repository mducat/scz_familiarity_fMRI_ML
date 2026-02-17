"""Utilities developed based on Nilearn
"""

import collections
import warnings

import numpy as np
import pandas as pd

from nilearn.image.resampling import coord_transform


def load_cut_coords(path):
    """Load data from csv and make coordinates to list of triplets

    Parameters
    ----------
    path : str
        Path to data.
        NOTE: only .csv files

    Returns
    -------
    coords : list
        List contains coordinates in triplets (x, y, z).

    data : pandas.DataFrame
        Loaded data from csv file
    """
    data = pd.read_csv(path)
    data = data.drop('Unnamed: 0', axis=1)

    coords = []
    for x, y, z in zip(data['x'], data['y'], data['z']):
        coord_ = (x, y, z)
        coords.append(coord_)

    return coords, data


def find_region_names(coords, atlas_img, labels=None):
    """Given list of MNI space coordinates, get names of the brain regions.

    Names of the brain regions are returned by getting nearest coordinates
    in the given `atlas_img` space iterated over the provided list of
    `coords`. These new image coordinates are then used to grab the label
    number (int) and name assigned to it. Last, these names are returned.

    Parameters
    ----------
    coords : Tuples of coordinates in a list
        MNI coordinates.

    atlas_img : Nifti-like image
        Path to or Nifti-like object. The labels (integers) ordered in
        this image should be sequential. Example: [0, 1, 2, 3, 4] but not
        [0, 5, 6, 7]. Helps in returning correct names without errors.

    labels : str in a list
        Names of the brain regions assigned to each label in atlas_img.
        NOTE: label with index 0 is assumed as background. Example:
            harvard oxford atlas. Hence be removed.

    Returns
    -------
    new_labels : int in a list
        Labels in integers generated according to correspondence with
        given atlas image and provided coordinates.

    names : str in a list
        Names of the brain regions generated according to given inputs.
    """
    affine = atlas_img.affine
    atlas_data = atlas_img.get_fdata()
    check_labels_from_atlas = np.unique(atlas_data)

    if labels is not None:
        names = []
        if not isinstance(labels, collections.Iterable):
            labels = np.asarray(labels)

    if isinstance(labels, collections.Iterable) and \
            isinstance(check_labels_from_atlas, collections.Iterable):
        if len(check_labels_from_atlas) != len(labels):
            warnings.warn("The number of labels provided does not match "
                          "with number of unique labels with atlas image.",
                          stacklevel=2)

    coords = list(coords)
    nearest_coordinates = []

    for sx, sy, sz in coords:
        nearest = np.round(coord_transform(sx, sy, sz, np.linalg.inv(affine)))
        nearest = nearest.astype(int)
        nearest = (nearest[0], nearest[1], nearest[2])
        nearest_coordinates.append(nearest)

    assert(len(nearest_coordinates) == len(coords))

    new_labels = []
    for coord_ in nearest_coordinates:
        # Grab index of current coordinate
        index = atlas_data[coord_]
        new_labels.append(index)
        if labels is not None:
            names.append(labels[index])

    if labels is not None:
        return new_labels, names
    else:
        return new_labels


""" A simple example to show how this function works.

####################################################################
# Grab atlas from Nilearn

from nilearn import datasets

atlas_name = 'cort-maxprob-thr0-1mm'
harvard = datasets.fetch_atlas_harvard_oxford(atlas_name=atlas_name)

#####################################################################
# Required inputs for function find_region_names_using_cut_coords

dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
atlas_img = harvard.maps
labels = harvard.labels

######################################################################
# Grab function

l, n = find_region_names_using_cut_coords(dmn_coords, atlas_img,
                                          labels=labels)

# where 'l' indicates new labels generated according to given atlas labels and
# coordinates
new_labels = l

# where 'n' indicates brain regions names labelled, generated according to given
# labels
region_names_involved = n

######################################################################
# Let's visualize
from nilearn.image import load_img
from nilearn import plotting
from nilearn.image import new_img_like

atlas_img = load_img(atlas_img)
affine = atlas_img.affine
atlas_data = atlas_img.get_data()

for i, this_label in enumerate(new_labels):
    this_data = (atlas_data == this_label)
    this_data = this_data.astype(int)
    this_img = new_img_like(atlas_img, this_data, affine)
    plotting.plot_roi(this_img, cut_coords=dmn_coords[i],
                      title=region_names_involved[i])

plotting.show()
"""
