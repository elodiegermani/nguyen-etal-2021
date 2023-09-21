"""Functions to extract labels corresponding to the atlas used for parcellation. 
Labels are based on the ROI of the AAL atlas that is the closest to each atlas' ROI centroid."""

import nibabel as nib
from nilearn import plotting, datasets
import matplotlib.pyplot as plt
import numpy as np

def apply_inv_matrix(
	affine:nib.Nifti1Image.affine, 
	coords:list
) -> list :
	"""Apply inverse matrix to get coordinates in the tabular space.

	Parameters
	----------

	affine : nib.Nifti1Image.affine
		affine to apply 
	coords : list of int
		coords in the MNI space 

	Returns
	-------

	list 
		coords in the tabular space
	"""
    inv_affine = np.linalg.inv(affine)
    
    M = inv_affine[:3, :3]
    abc = inv_affine[:3, 3]

    return M.dot(coords) + abc

def get_labels_from_aal(
	atlas: nilearn.Bunch, 
	map:bool=0
) -> list:
	"""Find the labels of the ROI corresponding to an atlas based on the closest AAL ROI value to the ROI centroid.

	Parameters
	----------

	atlas : nilearn.Bunch
		atlas to find labels for 
	map : bool
		whether the path element of the nilearn.Bunch is named "map" or "maps".

	Returns
	-------

	list 
		list of labels corresponding to each ROI. 
	"""
	# Load the atlas image
    if map:
        atlas_img = nib.load(atlas.map) 
    else: 
        atlas_img = nib.load(atlas.maps)

    centroid_coords = plotting.find_parcellation_cut_coords(atlas_img) # Centroid coords for each ROI 
    mm_coords = [[i for i in centroid_coords[j]] for j in range(len(centroid_coords))] # Coords in mm for the centroid of each ROI
    
    aal_atlas = datasets.fetch_atlas_aal()
    aal_img = nib.load(aal_atlas.maps)
    vox_coords = [apply_inv_matrix(aal_img.affine, c) for c in mm_coords] # Coords in tabular for the centroid of each ROI
    
    labels = []
    
    for i, c in enumerate(vox_coords):
    	# First look at the label at the exact centroid coord
        atlas_index = str(int(int(aal_img.get_fdata()[int(round(c[0],0))][int(round(c[1],0))][int(round(c[2],0))])))

        if atlas_index != '0': # If there is a label, find the corresponding name
            atlas_label = aal_atlas.labels[aal_atlas.indices.index(atlas_index)]

        else: # If not, look for any label in a growing square 
            n=1
            while atlas_index=='0' or atlas_index==float(0):
                sq = aal_img.get_fdata()[int(round(c[0],0))-n:int(round(c[0],
                                        0))+n, int(round(c[1],0))-n:int(round(c[1],0))+n, int(round(c[2],
                                        0))-n:int(round(c[2],0))+n]
                if np.any(sq!=float(0)):
                    atlas_index = str(int(sq[sq!=float(0)][0]))
                n+=1
            atlas_label = aal_atlas.labels[aal_atlas.indices.index(atlas_index)]

        labels.append(atlas_label)
    
    return labels

def get_atlas_labels():
	"""Function to automatise the search of labels for atlas used for parcellation.

	Write files in the './inputs/atlases' directory.
	"""
	basc197_atlas = datasets.fetch_atlas_basc_multiscale_2015(version="sym", resolution=197)
	basc444_atlas = datasets.fetch_atlas_basc_multiscale_2015(version="sym", resolution=444)
	schaefer_atlas = datasets.fetch_atlas_schaefer_2018(100)
	
	labels = []

	atlas_dict = {
	    'basc197':basc197_atlas,
	    'basc444': basc444_atlas,
	    'schaefer': schaefer_atlas
	}
	output_dir = './inputs/atlases'

	for atlas_name in atlas_dict.keys():
	    atlas = atlas_dict[atlas_name]
	    if atlas_name == 'schaefer':
	        map=0
	    else:
	        map=1
	    lab = get_labels_from_aal(atlas, map)
	    
	    if atlas_name == 'schaefer':
	        for i in range(28):
	            lab.append(f'Cerebellar_{i+1}')
	        for i in range(7):
	            lab.append(f'Striatum_{i+1}')
	            
	    with open(f'{output_dir}/{atlas_name}_labels.txt', 'w') as f:
	        for l in lab:
	            f.write(l)
	            f.write('\n')
	    f.close()