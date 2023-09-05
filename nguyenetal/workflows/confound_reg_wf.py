import glob, os
from os.path import join
from nipype.pipeline.engine import Workflow, Node, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
import nipype.interfaces.fsl as fsl

def get_regressors(confounds_file, confounds_list):
    import pandas as pd
    
    confounds_all = list(pd.read_csv(confounds_file, sep='\t').columns)
    idx_confounds_list = []
    
    for i, c in enumerate(confounds_all):
        if c in confounds_list: 
            idx_confounds_list.append(i+1)

    return idx_confounds_list

def filter_regressors(subject_list, data_dir, output_dir, confounds_list):
    workflow = Workflow('filter_reg', base_dir=os.path.join(output_dir, 'working_dir'))

    info_source = Node(IdentityInterface(fields=['subject_id']), name='info_source')
    info_source.iterables = [('subject_id', subject_list)]

    # Templates to select files node
    file_templates = {
        'func': join(
            'sub-{subject_id}', 'func', 'sub-{subject_id}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
            ), 
        'confounds': join(
            'sub-{subject_id}', 'func', 'sub-{subject_id}_task-rest_desc-confounds_timeseries.tsv'
        )
    }
    
    # SelectFiles node - to select necessary files
    select_files = Node(
        SelectFiles(file_templates, base_directory = data_dir),
        name='select_files'
    )

    reglist = Node(
        Function(function = get_regressors,
            input_names = ['confounds_file', 'confounds_list'],
            output_names = ['idx_confounds_list']), 
        name='reglist'
    )

    reglist.inputs.confounds_list = confounds_list
    
    regfilter = Node(fsl.FilterRegressor(), name='regfilter')

    # DataSink Node - store the wanted results in the wanted repository
    data_sink = Node(
        DataSink(base_directory = output_dir),
        name='data_sink',
    )

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