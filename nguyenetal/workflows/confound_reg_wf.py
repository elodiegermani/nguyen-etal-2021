import glob, os
from os.path import join
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
import nipype.interfaces.fsl as fsl

def get_regressors(confounds_file, confounds_list):
    """
    Get the list of column idx on the confounds file corresponding to the desired confounds list. 

    Parameters:
    - confounds_file : str, paths to subject parameters file 
    - confounds_list : list of str, list of columns names corresponding to existing columns in the confounds file

    Return :
    - idx_confounds_list : list of int, list of idx corresponding to columns in the confounds file. 
    """
    import pandas as pd

    confounds_all = list(pd.read_csv(confounds_file, sep='\t').columns)
    idx_confounds_list = []

    for i, c in enumerate(confounds_all):
        if c in confounds_list: 
            idx_confounds_list.append(i+1)

    assert(len(idx_confounds_list)==len(confounds_list))

    return idx_confounds_list

def get_affine_motion_param(confounds_file, subject_id, output_dir):
    """
    Create new tsv files with only desired parameters per subject per run.

    Parameters :
    - confounds_file : paths to subject parameters file 
    - subject_id : subject for whom the 1st level analysis is made
    - output_dir : path where to store new parameters file. 

    Return :
    - parameters_file : paths to new files containing only desired parameters.
    """
    from os import mkdir
    from os.path import join, isdir

    import pandas as pd
    import numpy as np

    # Handle the case where filepaths is a single path (str)
    if not isinstance(confounds_file, list):
        confounds_file = [confounds_file]

    # Create the parameters files
    parameters_file = []
    for file_id, file in enumerate(confounds_file):
        data_frame = pd.read_csv(file, sep = '\t', header=0)

        # Extract parameters we want to use for the model
        temp_list = np.array([
            data_frame['rot_x'], data_frame['rot_y'], data_frame['rot_z'],
            data_frame['trans_x'], data_frame['trans_y'], data_frame['trans_z']])
        retained_parameters = pd.DataFrame(np.transpose(temp_list))

        # Write parameters to a parameters file
        # TODO : warning !!! filepaths must be ordered (1,2,3,4) for the following code to work
        new_path =join(output_dir, 'parameters_file',
            f'parameters_file_sub-{subject_id}.tsv')

        if not isdir(join(output_dir, 'parameters_file')):
            mkdir(join(output_dir, 'parameters_file'))

        with open(new_path, 'w') as writer:
            writer.write(retained_parameters.to_csv(
                sep = '\t', index = False, header = False, na_rep = '0.0'))

        parameters_file.append(new_path)

    return parameters_file

class NoiseRegression_Pipeline:
    '''
    Class to create the noise regression pipeline. 

    Parameters:
    - subject_list : list of str, list of subjects to analyse.
    - data_dir : str, path to data directory
    - output_dir : str, path to output directory
    - confounds_list : list of str, noise regressors to filter from data
    - confounds_file_template : path, Nipype SelectFiles template for confounds files
    - func_file_template : path, Nipype SelectFiles template for confounds files
    - run_ICA : bool, whether to run ICA or not (NOT FUNCTIONAL FOR NOW IF SET TO YES)

    '''
    def __init__(self, subject_list, data_dir, output_dir, 
        confounds_list, confounds_file_template, func_file_template,
        run_ICA=False):

        self.subject_list = subject_list
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.confounds_list = confounds_list
        self.confounds_file_template = confounds_file_template
        self.func_file_template = func_file_template
        self.run_ICA = run_ICA

        self.pipeline = self.get_base_filter_regressors_pipeline()


    def get_filter_regressors_pipeline(self):
        '''
        Function to create Nipype workflow for noise regressors filtering if ICA is not performed. 
        '''

        workflow = Workflow('filter_reg', base_dir=os.path.join(self.output_dir, 'working_dir'))

        info_source = Node(IdentityInterface(fields=['subject_id']), name='info_source')
        info_source.iterables = [('subject_id', self.subject_list)]

        # Templates to select files node
        file_templates = {
            'func': self.func_file_template, 
            'confounds': self.confounds_file_template
        }
        
        # SelectFiles node - to select necessary files
        select_files = Node(
            SelectFiles(file_templates, base_directory = self.data_dir),
            name='select_files'
        )

        # Extract IDs of column of interest on the file. 
        reglist = Node(
            Function(function = get_regressors,
                input_names = ['confounds_file', 'confounds_list'],
                output_names = ['idx_confounds_list']), 
            name='reglist'
        )

        reglist.inputs.confounds_list = self.confounds_list
        
        regfilter = Node(fsl.FilterRegressor(output_type='NIFTI_GZ'), name='regfilter')

        # ICA-AROMA
        ica_aroma = Node(
            fsl.ICA_AROMA(out_dir=output_dir), 
            name='ica_aroma'
        )
        # ICA-AROMA only need the affine motion parameters. 
        motionparam = Node(
            Function(function = get_affine_motion_param,
                input_names = ['confounds_file', 'subject_id', 'output_dir'],
                output_names = ['parameters_file']), 
            name='motionparam'
        )

        # DataSink Node - store the wanted results in the wanted repository
        data_sink = Node(
            DataSink(base_directory = self.output_dir),
            name='data_sink',
        )
        data_sink.inputs.substitutions = [('_subject_id_', 'sub-')]

        if self.run_ICA == False: 
            # Solution without ICA
            workflow.connect(
                [
                    (info_source, select_files, [('subject_id', 'subject_id')]),
                    (select_files, reglist, [('confounds', 'confounds_file')]),
                    (select_files, regfilter, [('func', 'in_file'), 
                                              ('confounds', 'design_file')]),
                    (reglist, regfilter, [('idx_confounds_list', 'filter_columns')]),
                    (regfilter, data_sink, [('out_file', 'results.@filtered')])
                ]
            )

        else: 
            # Solution with ICA Included
            workflow.connect(
                [
                    (info_source, select_files, [('subject_id', 'subject_id')]),
                    (select_files, reglist, [('confounds', 'confounds_file')]),
                    (select_files, regfilter, [('func', 'in_file'), 
                                              ('confounds', 'design_file')]),
                    (reglist, regfilter, [('idx_confounds_list', 'filter_columns')]),
                    (regfilter, data_sink, [('out_file', 'results.@filtered')])
                ]
            )

        return workflow