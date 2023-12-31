import glob, os
from os.path import join
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
import nipype.interfaces.fsl as fsl

def get_ICA_motion_related_regressors(
    ica_directory : str, 
    subject_id : str):
    ''' 
    Creates the file containing only ICs classified as movement. ICA-AROMA outputs two different files: 
    - a txt file containing the IDs of the components classified as motion
    - a tsv file containing the timeseries of all components

    We extract a dataframe containing only the timeseries of the motion classified components. 

    Parameters
    ----------
    ica_directory : str
        Directory where all subjects ICA results are stored
    subject_id : str
        Idx of the subject to extract results

    Returns
    -------
    str 
        Path to the file containing regressors.
    '''
    from glob import glob
    import pandas as pd 
    from os.path import join 
    import os

    ica_idx_file = join(ica_directory, f'sub-{subject_id}', 'classified_motion_ICs.txt')
    ica_regressors_file = join(ica_directory, f'sub-{subject_id}', 'melodic.ica', 'melodic_mix')

    df_timeseries = pd.read_csv(ica_regressors_file, header=None, sep='\s+', index_col=None)

    with open(ica_idx_file, 'r') as f:
        ic_list = [int(i)-1 for i in f.read().split(',')] # List of motion-classified ICs

    df_motion_timeseries = df_timeseries.iloc[:,ic_list] 

    df_motion_timeseries_fpath = join(os.getcwd(), f'sub-{subject_id}_motion-related-ic.tsv')
    df_motion_timeseries.to_csv(df_motion_timeseries_fpath, 
        sep='\t', header = False, index = False)

    return df_motion_timeseries_fpath


def merge_confounds_files(
    motion_regressors_file : str, 
    ica_file : str, 
    wm_csf_file : str, 
    subject_id : str
) -> str:
    '''
    Merge all noise regressors to one single file.

    Parameters
    ----------
    motion_regressors_file : str
        filepath to motion regressors tsv file 

    ica_file : str
        filepath to IC-motion file

    wm_csf_file : str
        filepath to file containing wm and csf timeserie

    subject_id : str
        idx of the subject of interest

    Returns
    -------
    str 
        path to the merged tsv file
    '''
    import pandas as pd 
    from os.path import join 
    import os

    df_motion = pd.read_csv(motion_regressors_file, sep='\s+', index_col=None, header=None)
    df_ica = pd.read_csv(ica_file, sep='\s+', index_col = None, header = None)
    df_wm_csf = pd.read_csv(wm_csf_file, sep = '\s+', index_col = None)
    df_wm_csf = df_wm_csf[['white_matter', 'csf']]

    df = df_motion.merge(df_ica, left_index=True, right_index=True)
    df = df.merge(df_wm_csf, left_index=True, right_index=True)

    df_fpath = join(os.getcwd(), f'sub-{subject_id}_task-rest_desc-confounds_timeseries.tsv')

    df.to_csv(df_fpath, sep='\t', header=False, index=False)

    return df_fpath

class NoiseRegression_Pipeline:
    '''
    Class to create the noise regression pipeline. 

    Attributes
    ----------

    subject_list : list of str
        list of subjects to analyse

    data_dir : str
        path to data directory

    output_dir : str
        path to output directory

    wm_csf_template : str
        path to tsv file containing wm and csf time series (columns 'csf' and 'white_matter')

    motion_regressors_template : str
        path to tsv file containing 6 affine motion regressors 

    func_file_template : str
        Nipype SelectFiles template for confounds files

    include_ICA : bool
        whether to use ICA regressors or not. 

    ica_directory : str
        path to directory containing all subjects ICA outputs.

    pipeline : nipype.Workflow
        workflow to perform noise regression. 
    '''
    def __init__(self, 
        subject_list: list, 
        data_dir : str, 
        output_dir : str, 
        wm_csf_template : str, 
        motion_regressors_template : str, 
        func_file_template : str,
        include_ICA : bool =False, 
        ica_directory : str = ''
    ):
        """"
        Parameters
        ----------
        subject_list : list of str
            list of subjects to analyse

        data_dir : str
            path to data directory

        output_dir : str
            path to output directory

        wm_csf_template : str
            path to tsv file containing wm and csf time series (columns 'csf' and 'white_matter')

        motion_regressors_template : str
            path to tsv file containing 6 affine motion regressors 

        func_file_template : str
            Nipype SelectFiles template for confounds files

        include_ICA : bool
            whether to use ICA regressors or not. 

        ica_directory : str
            path to directory containing all subjects ICA outputs.
        """

        self.subject_list = subject_list
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.wm_csf_template = wm_csf_template
        self.motion_regressors_template = motion_regressors_template
        self.func_file_template = func_file_template
        self.include_ICA = include_ICA
        self.ica_directory = ica_directory

        self.pipeline = self.get_filter_regressors_pipeline()

    def get_filter_regressors_pipeline(self):
        '''
        Function to create Nipype workflow for noise regressors filtering if ICA is not performed. 

        Returns 
        -------
        nipype.Workflow
            workflow to perform noise regression. 
        '''

        workflow = Workflow('filter_reg', base_dir=os.path.join(self.output_dir, 'working_dir'))

        info_source = Node(IdentityInterface(fields=['subject_id']), name='info_source')
        info_source.iterables = [('subject_id', self.subject_list)]

        # Templates to select files node
        file_templates = {
            'func': self.func_file_template, 
            'motion_regressors': self.motion_regressors_template,
            'wm_csf': self.wm_csf_template
        }
        
        # SelectFiles node - to select necessary files
        select_files = Node(
            SelectFiles(file_templates, base_directory = self.data_dir),
            name='select_files'
        )

        ica_motion_regressors = Node(
            Function(input_names = ['ica_directory', 'subject_id'],
                output_names = ['df_motion_timeseries_fpath'],
                function = get_ICA_motion_related_regressors),
            name='ica_motion_regressors'
        )

        ica_motion_regressors.inputs.ica_directory = self.ica_directory

        merge_regressors = Node(
            Function(input_names=['motion_regressors_file', 'ica_file', 'wm_csf_file', 'subject_id'],
                output_names = ['df_fpath'],
                function = merge_confounds_files),
            name = 'merge_regressors'
        )

        regfilter = Node(fsl.FilterRegressor(output_type='NIFTI_GZ',
            filter_all=True), name='regfilter')

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(
            DataSink(base_directory = self.output_dir),
            name='data_sink',
        )
        data_sink.inputs.substitutions = [('_subject_id_', 'sub-')]

        # Connect nodes
        workflow.connect(
            [
                (info_source, select_files, [('subject_id', 'subject_id')]),
                (info_source, ica_motion_regressors, [('subject_id', 'subject_id')]),
                (select_files, regfilter, [('func', 'in_file')]),
                (select_files, merge_regressors, [('motion_regressors', 'motion_regressors_file'),
                                                ('wm_csf', 'wm_csf_file')]),
                (ica_motion_regressors, merge_regressors, [('df_motion_timeseries_fpath', 'ica_file')]),
                (info_source, merge_regressors, [('subject_id', 'subject_id')]),
                (merge_regressors, data_sink, [('df_fpath', 'denoising.@confounds')]),
                (merge_regressors, regfilter, [('df_fpath', 'design_file')]),
                (regfilter, data_sink, [('out_file', 'denoising.@filtered')])
            ]
        )

        return workflow