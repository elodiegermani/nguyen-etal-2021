import glob, os
from os.path import join
from nilearn import image as nilimage
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype import IdentityInterface, Rename, DataSink, Function
from nipype.interfaces.fsl import ExtractROI, Merge, ImageStats, ImageMaths
from ..utils import alff, reho
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink


def get_mean_ROI_values(
    feature_image : str, 
    feature : str, 
    subject_id : str, 
    cereb_atlas : str, 
    striatum_atlas : str
) -> list:
    """Computes mean value per ROI for an image based on an atlas parcellation.

    Parameters
    ----------
    feature_image : str
        path to the image containing features to average on ROIs

    feature : str
        which feature (fALFF, ALFF or ReHo)

    subject_id : str
        idx of the subject of interest

    cereb_atlas : str 
        path to cerebellar atlas 

    striatum_atlas : str
        path to striatal atlas

    Returns
    -------
    list 
        List of paths to files containing the mean values per ROI for each atlas. 
    
    """
    from nilearn import regions, image, datasets
    import numpy as np
    import pandas as pd
    import os

    basc_atlas = datasets.fetch_atlas_basc_multiscale_2015()
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    
    # Paths to brain atlases
    atlas_dict = {'basc197': basc_atlas['scale197'],
                   'basc444': basc_atlas['scale444'],
                   'schaefer': [schaefer_atlas, cereb_atlas, striatum_atlas]
                   }

    df_list = []
    
    for atlas in atlas_dict.keys():
        atlas_path = atlas_dict[atlas]

        if atlas == 'schaefer': # First compute for Schaefer ROI and then for Cerebellar and Striatal
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

def get_zscore(
    feature_image : str,
    mask : str,
    ):
    '''
    Function to convert to Z-Score images.

    Parameters
    ----------
    feature_image : str
        path to the image containing features to average on ROIs

    mask : str
        path to the mask image

    feature : str
        which feature (fALFF, ALFF or ReHo)

    subject_id : str
        idx of the subject of interest

    Returns
    -------
    file : str
        path to z-score file
    '''

    mean_val = Node(ImageStats(
        in_file = feature_image, 
        op_string = '-m', 
        mask_file = mask), 
        name = 'mean_val')

    mean = mean_val.run().outputs.out_stat

    std_val = Node(ImageStats(
        in_file = feature_image, 
        op_string = '-s', 
        mask_file = mask), 
        name = 'mean_val')

    std = std_val.run().outputs.out_stat

    normalize = Node(ImageMaths(
        in_file = feature_image,
        mask_file = mask,
        op_string = f'-sub {mean} -div {std}'), 
        name = 'normalize')

    z_score_image = normalize.run().outputs

    return z_score_image

class StaticMeasures_Pipeline:
    '''
    Class to build pipeline to compute static ALFF, fALFF, and ReHo for an fMRI image.

    Attributes
    ----------
    subject_list : list of str
        list of subjects to analyse.

    data_dir : str
        path to data directory

    output_dir : str
        path to output directory

    mask_file_template : str
        Nipype SelectFiles template for mask files

    func_file_template : str
        Nipype SelectFiles template for confounds files

    high_pass_filter : float
        threshold for high pass filtering

    low_pass_filter : float
        threshold for low pass filtering

    cluster_size : int
        number of neighbor to take into account for ReHo computation

    pipeline : nipype.Workflow
        workflow to compute static ALFF, fALFF and ReHO for an image
    '''

    def __init__(self, 
        subject_list: list, 
        data_dir : str, 
        output_dir : str, 
        mask_file_template : str, 
        func_file_template : str, 
        cereb_atlas : str, 
        striatum_atlas : str,
        high_pass_filter : float, 
        low_pass_filter : float, 
        cluster_size : int
    ):
        """
        Parameters
        ----------

        subject_list : list of str
            list of subjects to analyse.

        data_dir : str
            path to data directory

        output_dir : str
            path to output directory

        mask_file_template : str
            Nipype SelectFiles template for mask files

        func_file_template : str
            Nipype SelectFiles template for confounds files

        high_pass_filter : float
            threshold for high pass filtering

        low_pass_filter : float
            threshold for low pass filtering

        cluster_size : int
            number of neighbor to take into account for ReHo computation
        """

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
        """Creates the workflow to compute ALFF, ReHo and fALFF on images. 

        Returns
        -------
        nipype.Workflow
            workflow to compute ALFF, ReHo and fALFF on images.

        """
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

        # Normalize images
        normalize_wf_alff = Node(Function(
            function=get_zscore,
            input_names = ['feature_image', 'mask'],
            output_names = ['z_score_image']), 
        name='normalize_wf_alff')

        normalize_wf_falff = Node(Function(
            function=get_zscore,
            input_names = ['feature_image', 'mask'],
            output_names = ['z_score_image']), 
        name='normalize_wf_falff')

        normalize_wf_reho = Node(Function(
            function=get_zscore,
            input_names = ['feature_image', 'mask'],
            output_names = ['z_score_image']), 
        name='normalize_wf_reho')

        # Compute mean regional values
        compute_roi_measures_alff = Node(
            Function(function=get_mean_ROI_values, 
                input_names=['feature_image', 'feature', 'subject_id', 'cereb_atlas', 'striatum_atlas'],
                output_names=['df_list']), 
            name='compute_roi_measures_alff')

        compute_roi_measures_alff.inputs.feature = 'alff'
        compute_roi_measures_alff.inputs.cereb_atlas = self.cereb_atlas
        compute_roi_measures_alff.inputs.striatum_atlas = self.striatum_atlas

        compute_roi_measures_falff = Node(
            Function(function=get_mean_ROI_values, 
                input_names=['feature_image', 'feature', 'subject_id', 'cereb_atlas', 'striatum_atlas'],
                output_names=['df_list']), 
            name='compute_roi_measures_falff')

        compute_roi_measures_falff.inputs.feature = 'falff'
        compute_roi_measures_falff.inputs.cereb_atlas = self.cereb_atlas
        compute_roi_measures_falff.inputs.striatum_atlas = self.striatum_atlas

        compute_roi_measures_reho = Node(
            Function(function=get_mean_ROI_values, 
                input_names=['feature_image', 'feature', 'subject_id','cereb_atlas', 'striatum_atlas'],
                output_names=['df_list']), 
            name='compute_roi_measures_reho')

        compute_roi_measures_reho.inputs.feature = 'ReHo'
        compute_roi_measures_reho.inputs.cereb_atlas = self.cereb_atlas
        compute_roi_measures_reho.inputs.striatum_atlas = self.striatum_atlas

        compute_roi_measures_zalff = Node(
            Function(function=get_mean_ROI_values, 
                input_names=['feature_image', 'feature', 'subject_id', 'cereb_atlas', 'striatum_atlas'],
                output_names=['df_list']), 
            name='compute_roi_measures_zalff')

        compute_roi_measures_alff.inputs.feature = 'zalff'
        compute_roi_measures_alff.inputs.cereb_atlas = self.cereb_atlas
        compute_roi_measures_alff.inputs.striatum_atlas = self.striatum_atlas

        compute_roi_measures_zfalff = Node(
            Function(function=get_mean_ROI_values, 
                input_names=['feature_image', 'feature', 'subject_id', 'cereb_atlas', 'striatum_atlas'],
                output_names=['df_list']), 
            name='compute_roi_measures_zfalff')

        compute_roi_measures_falff.inputs.feature = 'zfalff'
        compute_roi_measures_falff.inputs.cereb_atlas = self.cereb_atlas
        compute_roi_measures_falff.inputs.striatum_atlas = self.striatum_atlas

        compute_roi_measures_zreho = Node(
            Function(function=get_mean_ROI_values, 
                input_names=['feature_image', 'feature', 'subject_id','cereb_atlas', 'striatum_atlas'],
                output_names=['df_list']), 
            name='compute_roi_measures_zreho')

        compute_roi_measures_reho.inputs.feature = 'zReHo'
        compute_roi_measures_reho.inputs.cereb_atlas = self.cereb_atlas
        compute_roi_measures_reho.inputs.striatum_atlas = self.striatum_atlas

        # Copy outputs into a user-friendly location
        datasink = Node(DataSink(base_directory=self.output_dir, remove_dest_dir=True), name='datasink')
        datasink.inputs.substitutions = [('_subject_id_', 'sub-')]

        workflow = Workflow('static_measures_wf', base_dir=os.path.join(self.output_dir, 'working_dir'))

        workflow.connect(info_source, 'subject_id', select_files, 'subject_id')
        workflow.connect(select_files, 'mask', alff_workflow, 'inputspec.rest_mask')
        workflow.connect(select_files, 'mask', reho_workflow, 'inputspec.rest_mask')
        workflow.connect(select_files, 'func', alff_workflow, 'inputspec.rest_res')
        workflow.connect(select_files, 'func', reho_workflow, 'inputspec.rest_res_filt')
        workflow.connect(alff_workflow, 'outputspec.alff_img', datasink, 'features.@alff')
        workflow.connect(alff_workflow, 'outputspec.falff_img', datasink, 'features.@falff')
        workflow.connect(reho_workflow, 'outputspec.raw_reho_map', datasink, 'features.@reho')
        
        # Normalized images
        workflow.connect([
            (alff_workflow, normalize_wf_alff, 
                [('outputspec.alff_img', 'feature_image')]),
            (select_files, normalize_wf_alff, 
                [('mask', 'mask')]),
            (alff_workflow, normalize_wf_falff, 
                [('outputspec.falff_img', 'feature_image')]),
            (select_files, normalize_wf_falff, 
                [('mask', 'mask')]),
            (alff_workflow, normalize_wf_reho, 
                [('outputspec.raw_reho_map', 'feature_image')]),
            (select_files, normalize_wf_reho, 
                [('mask', 'mask')]),
            (info_source, compute_roi_measures_zalff, 
                [('subject_id', 'subject_id')]),
            (normalize_wf_alff, compute_roi_measures_zalff, 
                [('z_score_image', 'feature_image')]),
            (info_source, compute_roi_measures_zfalff, 
                [('subject_id', 'subject_id')]),
            (normalize_wf_falff, compute_roi_measures_zfalff,
                [('z_score_image', 'feature_image')]),
            (info_source, compute_roi_measures_zreho, 
                [('subject_id', 'subject_id')]),
            (normalize_wf_reho, compute_roi_measures_zreho, 
                [('z_score_image', 'feature_image')]),
            (compute_roi_measures_zalff, datasink, 
                [('df_list', 'features.@zalff_atlas')]),
            (compute_roi_measures_zfalff, datasink, 
                [('df_list', 'features.@zfalff_atlas')]),
            (compute_roi_measures_zreho, datasink, 
                [('df_list', 'features.@zreho_atlas')])
            ])

        # Not normalized images
        workflow.connect([
            (alff_workflow, compute_roi_measures_alff, 
                [('outputspec.alff_img', 'feature_image')]),
            (alff_workflow, compute_roi_measures_falff, 
                [('outputspec.falff_img', 'feature_image')]),
            (alff_workflow, compute_roi_measures_reho, 
                [('outputspec.raw_reho_map', 'feature_image')]),
            (info_source, compute_roi_measures_alff, 
                [('subject_id', 'subject_id')]),
            (info_source, compute_roi_measures_falff, 
                [('subject_id', 'subject_id')]),
            (info_source, compute_roi_measures_reho, 
                [('subject_id', 'subject_id')]),
            (compute_roi_measures_alff, datasink, 
                [('df_list', 'features.@alff_atlas')]),
            (compute_roi_measures_falff, datasink, 
                [('df_list', 'features.@falff_atlas')]),
            (compute_roi_measures_reho, datasink, 
                [('df_list', 'features.@reho_atlas')])
            ])


        return workflow