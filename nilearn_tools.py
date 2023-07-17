#
# Compute functional connectomes; loosely based on Nilearn Example 03, see
# https://nilearn.github.io/stable/auto_examples/03_connectivity/plot_inverse_covariance_connectome.html#compute-the-sparse-inverse-covariance
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from nilearn.maskers import NiftiSpheresMasker
from sklearn.covariance import GraphicalLassoCV
from pydantic import validate_arguments

# Remove subjects for which partial correlation computation diverges
subjectList = list(set(range(33)).difference([7,8,11,12,19,21,22,23,24,26,27,28]))

@validate_arguments
def compute_connectome(
        subidx : int,
        outfolder: str
    ) -> np.ndarray:
    """
    Compute functional connectome of single subject

    Parameters
    ----------
    subidx : int
        Subject number
    outfolder : str
        Path to a directory for storing generated connectome renderings
        (must already exist)

    Returns
    -------
    con : 2D np.ndarray
        Functional connectivity matrix
    """

    # Take stock of data on disk
    data = datasets.fetch_development_fmri(age_group="adult",
                                           data_dir="/cs/home/fuertingers/nilearn_data")
    atlas = datasets.fetch_coords_power_2011(legacy_format=False)
    atlasCoords = np.vstack((atlas.rois['x'], atlas.rois['y'], atlas.rois['z'])).T

    # Extract fMRI time-series averaged within spheres @ atlas coords
    masker = NiftiSpheresMasker(seeds=atlasCoords, smoothing_fwhm=6, radius=5., detrend=True, standardize=True,
                                low_pass=0.1, high_pass=0.01, t_r=2)
    timeseries = masker.fit_transform(data.func[subidx], confounds=data.confounds[subidx])

    # Compute functional connectivity b/w brain regions
    estimator = GraphicalLassoCV()
    estimator.fit(timeseries)
    con = estimator.covariance_

    # Save connectome plot
    fig = plt.figure()
    plotting.plot_connectome(con, atlasCoords, title=f"Subject #{subidx}",
                             edge_threshold="95%", node_size=20, colorbar=True,
                             edge_vmin=-1, edge_vmax=1, figure=fig)
    fig.savefig(os.path.join(outfolder, f"subject{subidx}.png"))

    return con
