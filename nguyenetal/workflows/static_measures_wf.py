import glob, os
from os.path import join
from nilearn import image as nilimage
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype import IdentityInterface, Rename, DataSink
from nipype.interfaces.fsl import ExtractROI, Merge
from ..utils import alff, reho
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink

def static_measures(subject_list, data_dir, output_dir,
                    high_pass_filter=0.01, low_pass_filter=0.1, cluster_size=27):
    '''
    Compute static ALFF, fALFF, and ReHo for an fMRI image.

    :param func_image: path to preprocessed (normalized, denoised, etc.) fMRI NIfTI file
    :type func_image: str
    :param mask_image: path to binary brain mask NIfTI file
    :type mask_iamge: str
    :param output_dir: path to output directory
    :type output_dir: str
    :param high_pass_filter: cutoff in Hz for highpass filter. Default 0.01 Hz
    :type high_pass_filter: float
    :param low_pass_filter: cutoff in Hz for lowpass filter. Default 0.1 Hz
    :type low_pass_filter: float
    :param cluster_size: cluster size (neighborhood) for ReHo computation. Can be 7, 19, or 27. Default 27
    :type cluster_size: int
    :param n_job: number of parallel processes
    :type n_job: int
    '''

    output_dir = os.path.realpath(output_dir)

    if cluster_size not in [7, 19, 27]:
        raise ValueError('{} is not a valid cluster size. Must be 7, 19, or 27'.format(cluster_size))

    workflow = Workflow('static', base_dir=os.path.join(output_dir, 'working_dir'))

    info_source = Node(IdentityInterface(fields=['subject_id']), name='info_source')
    info_source.iterables = [('subject_id', subject_list)]

    # Templates to select files node
    file_templates = {
        'func': join(
            'sub-{subject_id}', 'func', 'sub-{subject_id}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
            ), 
        'mask': join(
            'sub-{subject_id}', 'func', 'sub-{subject_id}_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
        )
    }
    
    # SelectFiles node - to select necessary files
    select_files = Node(
        SelectFiles(file_templates, base_directory = data_dir),
        name='select_files'
    )

    # Compute ALFF and fALFF with CPAC's workflow
    alff_workflow = alff.create_alff('alff_workflow')
    alff_workflow.inputs.hp_input.hp = high_pass_filter
    alff_workflow.inputs.lp_input.lp = low_pass_filter

    # Compute ReHo with CPAC's workflow
    reho_workflow = reho.create_reho('reho_workflow')
    reho_workflow.inputs.inputspec.cluster_size = cluster_size

    # Copy outputs into a user-friendly location
    datasink = Node(DataSink(base_directory=output_dir, remove_dest_dir=True), name='datasink')
    workflow.connect(info_source, 'subject_id', select_files, 'subject_id')
    workflow.connect(select_files, 'mask', alff_workflow, 'inputspec.rest_mask')
    workflow.connect(select_files, 'mask', reho_workflow, 'inputspec.rest_mask')
    workflow.connect(select_files, 'func', alff_workflow, 'inputspec.rest_res')
    workflow.connect(select_files, 'func', reho_workflow, 'inputspec.rest_res_filt')
    workflow.connect(alff_workflow, 'outputspec.alff_img', datasink, 'results.@alff')
    workflow.connect(alff_workflow, 'outputspec.falff_img', datasink, 'results.@falff')
    workflow.connect(reho_workflow, 'outputspec.raw_reho_map', datasink, 'results.@reho')

    return workflow