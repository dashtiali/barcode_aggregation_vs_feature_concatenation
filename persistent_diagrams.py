"""
Computation of persistent diagrams
@author: Dashti Ali
"""

import numpy as np
from gudhi import CubicalComplex, AlphaComplex
from ripser import ripser
from enums import FiltrationTypes

def compute_pds(data, max_dim=1, filtration_type=FiltrationTypes.CUBICAL_COMPLEX):
    pds = []

    if filtration_type == FiltrationTypes.CUBICAL_COMPLEX:
        cub_filtration = CubicalComplex(dimensions=data.shape, top_dimensional_cells=data.flatten('F'))
        cub_filtration.persistence()

        for i in range(max_dim + 1):  
            ph_temp = cub_filtration.persistence_intervals_in_dimension(i)
            ph_temp = ph_temp[~np.isinf(ph_temp).any(axis=1),:]
            pds.append(ph_temp)

    elif filtration_type == FiltrationTypes.LOWER_STAR:
        raise NotImplementedError("This functionality is not implemented yet.")
    
    return pds


def compute_pds_from_point_cloud(data, max_dim=1, sub_samp_ratio=None, filtration_type=FiltrationTypes.ALPHA_COMPLEX, is_distance_matrix=False):
    pds = []

    if data.shape == 0:
        return pds
    
    if filtration_type == FiltrationTypes.ALPHA_COMPLEX:
        alpha_filtration = AlphaComplex(points = data).create_simplex_tree()
        alpha_filtration.persistence()

        for i in range(max_dim + 1):  
            ph_temp = alpha_filtration.persistence_intervals_in_dimension(i)
            ph_temp = ph_temp[~np.isinf(ph_temp).any(axis=1),:]
            pds.append(ph_temp)
    
    elif filtration_type == FiltrationTypes.LOWER_STAR:
        raise NotImplementedError("This functionality is not implemented yet.")

    elif filtration_type == FiltrationTypes.RIPS_COMPLEX:
        if sub_samp_ratio is not None:
            n_perm = int((data.shape[0])/sub_samp_ratio)
            dgms = ripser(data, maxdim=max_dim, n_perm=n_perm, distance_matrix=is_distance_matrix)['dgms']

            for ph in dgms:
                pds.append(ph[~np.isinf(ph).any(axis=1),:])
        else:
            dgms = ripser(data, maxdim=max_dim, distance_matrix=is_distance_matrix)['dgms']

            for ph in dgms:
                pds.append(ph[~np.isinf(ph).any(axis=1),:])

    return pds