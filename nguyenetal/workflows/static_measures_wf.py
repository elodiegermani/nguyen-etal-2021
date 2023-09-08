import glob, os
from os.path import join
from nilearn import image as nilimage
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype import IdentityInterface, Rename, DataSink, Function
from nipype.interfaces.fsl import ExtractROI, Merge
from ..utils import alff, reho
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink


def get_mean_ROI_values(feature_image, feature, subject_id, cereb_atlas, striatum_atlas):
    from nilearn import regions, image, datasets
    import numpy as np
    import pandas as pd
    import os

    basc_atlas = datasets.fetch_atlas_basc_multiscale_2015()
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    #cereb_atlas = './inputs/atlases/Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz'
    #striatum_atlas = './inputs/atlases/striatum-con-label-thr25-7sub-2mm.nii.gz'
    
    # Paths to brain atlases
    atlas_dict = {'basc197': basc_atlas['scale197'],
                   'basc444': basc_atlas['scale444'],
                   'schaefer': [schaefer_atlas, cereb_atlas, striatum_atlas]
                   }

    df_list = []
    
    for atlas in atlas_dict.keys():
        atlas_path = atlas_dict[atlas]

        if atlas == 'schaefer':
            output_list = []
            atlas_img = image.load_img(atlas_path[0]['maps'])
            cereb_atlas_img = image.load_img(atlas_path[1])
            striatum_atlas_img = image.load_img(atlas_path[2])
            
            atlas_img_res = image.resample_to_img(atlas_img, feature_image, 
                                                  interpolation='nearest')
            arr_regions, _ = regions.img_to_signals_labels([feature_image], atlas_img_res)
            
            cereb_atlas_img_res = image.resample_to_img(cereb_atlas_img, feature_image, 
                                                        interpolation='nearest')
            arr_regions_cereb, _ = regions.img_to_signals_labels([feature_image], cereb_atlas_img_res)
            
            striatum_atlas_img_res = image.resample_to_img(striatum_atlas_img, feature_image, 
                                                           interpolation='nearest')
            arr_regions_striatum, _ = regions.img_to_signals_labels([feature_image], striatum_atlas_img_res)
    
            arr_cort_cereb = np.append(arr_regions[0], arr_regions_cereb[0])
            arr_cort_cereb_striatum = np.append(arr_cort_cereb, arr_regions_striatum[0])

            output_list.append(arr_cort_cereb_striatum)
                
        else:
            output_list = []
            atlas_img = image.load_img(atlas_path)

            atlas_img_res = image.resample_to_img(atlas_img, feature_image, interpolation='nearest')
            arr_regions, _ = regions.img_to_signals_labels([feature_image], atlas_img_res)
            
            output_list.append(arr_regions[0])

        df = pd.DataFrame(output_list)
        basedir = os.path.dirname(feature_image)
        filename = f'{basedir}/sub-{subject_id}_{feature}_atlas-{atlas}.csv'
        df.to_csv(filename)
        df_list.append(filename)
    
    return df_list

class StaticMeasures_Pipeline:
    '''
    Class to build pipeline to compute static ALFF, fALFF, and ReHo for an fMRI image.

    Parameters:
    - subject_list : list of str, list of subjects to analyse.
    - data_dir : str, path to data directory
    - output_dir : str, path to output directory
    - mask_file_template : path, Nipype SelectFiles template for mask files
    - func_file_template : path, Nipype SelectFiles template for confounds files
    - high_pass_filter : float, threshold for high pass filtering
    - low_pass_filter : float, threshold for low pass filtering
    - cluster_size : int, number of neighbor to take into account for ReHo computation
    '''

    def __init__(self, subject_list, data_dir, output_dir, 
        mask_file_template, func_file_template, cereb_atlas, striatum_atlas,
        high_pass_filter, low_pass_filter, cluster_size):

        self.subject_list = subject_list
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.mask_file_template = mask_file_template
        self.func_file_template = func_file_template
        self.high_pass_filter = high_pass_filter
        self.low_pass_filter = low_pass_filter 
        self.cluster_size = cluster_size
        self.cereb_atlas = cereb_atlas
        self.striatum_atlas = striatum_atlas

        self.pipeline = self.get_static_measures_pipeline()

    def get_static_measures_pipeline(self):
        output_dir = os.path.realpath(self.output_dir)

        if self.cluster_size not in [7, 19, 27]:
            raise ValueError('{} is not a valid cluster size. Must be 7, 19, or 27'.format(self.cluster_size))

        info_source = Node(IdentityInterface(fields=['subject_id']), name='info_source')
        info_source.iterables = [('subject_id', self.subject_list)]

        # Templates to select files node
        file_templates = {
            'func': self.func_file_template, 
            'mask': self.mask_file_template
        }
        
        # SelectFiles node - to select necessary files
        select_files = Node(
            SelectFiles(file_templates, base_directory = self.data_dir),
            name='select_files'
        )

        # Compute ALFF and fALFF with CPAC's workflow
        alff_workflow = alff.create_alff('alff_workflow')
        alff_workflow.inputs.hp_input.hp = self.high_pass_filter
        alff_workflow.inputs.lp_input.lp = self.low_pass_filter

        # Compute ReHo with CPAC's workflow
        reho_workflow = reho.create_reho('reho_workflow')
        reho_workflow.inputs.inputspec.cluster_size = self.cluster_size

        compute_roi_measures_alff = Node(
            Function(function=get_mean_ROI_values, 
                input_names=['feature_image', 'feature', 'subject_id', 'cereb_atlas', 'striatum_atlas'],
                output_names=['df_list']), 
            name='compute_roi_measures_alff')

        compute_roi_measures_falff = Node(
            Function(function=get_mean_ROI_values, 
                input_names=['feature_image', 'feature', 'subject_id', 'cereb_atlas', 'striatum_atlas'],
                output_names=['df_list']), 
            name='compute_roi_measures_falff')

        compute_roi_measures_reho = Node(
            Function(function=get_mean_ROI_values, 
                input_names=['feature_image', 'feature', 'subject_id','cereb_atlas', 'striatum_atlas'],
                output_names=['df_list']), 
            name='compute_roi_measures_reho')

        compute_roi_measures_reho.inputs.feature = 'ReHo'
        compute_roi_measures_reho.inputs.cereb_atlas = self.cereb_atlas
        compute_roi_measures_reho.inputs.striatum_atlas = self.striatum_atlas

        # Copy outputs into a user-friendly location
        datasink = Node(DataSink(base_directory=self.output_dir, remove_dest_dir=True), name='datasink')
        datasink.inputs.substitutions = [('_subject_id_', 'sub-')]

        workflow = Workflow('static_measures_wf', base_dir=self.output_dir)

        workflow.connect(info_source, 'subject_id', select_files, 'subject_id')
        workflow.connect(select_files, 'mask', alff_workflow, 'inputspec.rest_mask')
        workflow.connect(select_files, 'mask', reho_workflow, 'inputspec.rest_mask')
        workflow.connect(select_files, 'func', alff_workflow, 'inputspec.rest_res')
        workflow.connect(select_files, 'func', reho_workflow, 'inputspec.rest_res_filt')
        workflow.connect(alff_workflow, 'outputspec.alff_img', datasink, 'results.@alff')
        workflow.connect(alff_workflow, 'outputspec.falff_img', datasink, 'results.@falff')
        workflow.connect(reho_workflow, 'outputspec.raw_reho_map', datasink, 'results.@reho')
        
        workflow.connect([
            (info_source, compute_roi_measures_alff, 
                [('subject_id', 'subject_id')]),
            (alff_workflow, compute_roi_measures_alff, 
                [('outputspec.alff_img', 'feature_image')]),
            (info_source, compute_roi_measures_falff, 
                [('subject_id', 'subject_id')]),
            (alff_workflow, compute_roi_measures_falff,
                [('outputspec.falff_img', 'feature_image')]),
            (info_source, compute_roi_measures_reho, 
                [('subject_id', 'subject_id')]),
            (reho_workflow, compute_roi_measures_reho, 
                [('outputspec.raw_reho_map', 'feature_image')]),
            (compute_roi_measures_alff, datasink, 
                [('df_list', 'results.@alff_atlas')])
            (compute_roi_measures_falff, datasink, 
                [('df_list', 'results.@falff_atlas')])
            (compute_roi_measures_reho, datasink, 
                [('df_list', 'results.@reho_atlas')])
            ])


        return workflow