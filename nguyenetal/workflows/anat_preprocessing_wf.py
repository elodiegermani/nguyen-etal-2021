import glob, os
from os.path import join
from nipype.pipeline.engine import Workflow, Node, JoinNode, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
from nipype.algorithms.misc import Gunzip

def get_timeseries_confounds_file(file_list, subject_id):
	import pandas as pd
	from os.path import join
	import os 
	filename = join(os.getcwd(), f'{subject_id}_task-rest_desc-confounds_timeseries.tsv')
	df_list = [pd.read_csv(f, sep='\s+', header=None, index_col=None) for f in sorted(file_list)]

	df = pd.DataFrame({'csf':df_list[0][0].tolist(), 'white_matter': df_list[-1][0].tolist()})

	df.to_csv(filename, header=True, sep='\t', index=False)

	return filename


class Anatomical_Preprocessing:
	'''
	Class to create the anatomical preprocessing pipeline. 

	Parameters:
	- subject_list : list of str, list of subjects to analyse.
	- data_dir : str, path to data directory
	- anat_file_template : path, Nipype SelectFiles template for anatomic files
	- output_dir : str, path to output directory
	- software = str, FSL or AFNI 
	'''

	def __init__(self, subject_list, data_dir, anat_file_template, func_file_template, output_dir, software):
		self.subject_list = subject_list
		self.data_dir = data_dir
		self.anat_file_template = anat_file_template
		self.func_file_template = func_file_template
		self.output_dir = output_dir
		self.software = software

		if self.software=='fsl':
			self.pipeline = self.get_anat_preproc_fsl_wf()
		elif self.software=='afni':
			self.pipeline = self.get_anat_preproc_afni_wf()

	def get_anat_preproc_fsl_wf(self):
		'''
		Function to create Nipype workflow for extraction of WM and CSF time series. 
		'''

		workflow = Workflow('anat_preproc_wf_fsl', base_dir=os.path.join(self.output_dir, 'working_dir'))

		info_source = Node(IdentityInterface(fields=['subject_id']), name='info_source')
		info_source.iterables = [('subject_id', self.subject_list)]

		# Templates to select files node
		file_templates = {
			'anat': self.anat_file_template, 
			'func': self.func_file_template
		}
		
		# SelectFiles node - to select necessary files
		select_files = Node(
			SelectFiles(file_templates, base_directory = self.data_dir),
			name='select_files'
		)

		brain_extraction = Node(
			fsl.BET(),
			name='brain_extraction'
		)

		segment = Node(
			fsl.FAST(segments=True, number_classes=3),
			name='segment'
		)

		resample = MapNode(
			fsl.FLIRT(apply_xfm=True, uses_qform=True),
			name='resample',
			iterfield=['in_file']
		)

		extract_signal = MapNode(
			fsl.ImageMeants(),
			name='extract_signal',
			iterfield=['mask'])

		create_df = Node(
			Function(input_names=['file_list', 'subject_id'],
					output_names=['filename'],
					function=get_timeseries_confounds_file),
			name = 'create_df')

		# DataSink Node - store the wanted results in the wanted repository
		data_sink = Node(
			DataSink(base_directory = self.output_dir),
			name='data_sink',
		)

		data_sink.inputs.substitutions = [('_subject_id_', 'sub-')]

		workflow.connect([
			(info_source, select_files, [('subject_id', 'subject_id')]),
			(select_files, brain_extraction, [('anat', 'in_file')]),
			(brain_extraction, segment, [('out_file', 'in_files')]),
			(segment, resample, [('tissue_class_files', 'in_file')]),
			(select_files, resample, [('func', 'reference')]),
			(resample, extract_signal, [('out_file', 'mask')]),
			(select_files, extract_signal, [('func', 'in_file')]),
			(extract_signal, data_sink, [('out_file', 'results_fsl.@1D_TS')]),
			(extract_signal, create_df, [('out_file', 'file_list')]),
			(info_source, create_df, [('subject_id', 'subject_id')]),
			(create_df, data_sink, [('filename', 'results_fsl.@df')])
		])

		return workflow


	def get_anat_preproc_afni_wf(self):
		'''
		Function to create Nipype workflow for extraction of WM and CSF time series. 
		'''
		workflow = Workflow('anat_preproc_wf_afni', base_dir=os.path.join(self.output_dir, 'working_dir'))

		info_source = Node(IdentityInterface(fields=['subject_id']), name='info_source')
		info_source.iterables = [('subject_id', self.subject_list)]

		# Templates to select files node
		file_templates = {
			'anat': self.anat_file_template, 
			'func': self.func_file_template
		}
		
		# SelectFiles node - to select necessary files
		select_files = Node(
			SelectFiles(file_templates, base_directory = self.data_dir),
			name='select_files'
		)

		brain_extraction = Node(
			afni.SkullStrip(),
			name='brain_extraction'
		)

		segment = Node(
			afni.Seg(mask='AUTO'),
			name='segment'
		)

		extract_maps = Node(
			afni.Calc(outputtype='NIFTI_GZ'),
			name = 'extract_maps'
		)

		extract_maps.iterables = [("expr", ['equals(a, 1)', 'equals(a, 2)', 'equals(a, 3)']),
									("out_file", ['1.nii.gz', '2.nii.gz', '3.nii.gz'])]
		extract_maps.synchronize = True


		resample = Node(
			afni.Resample(outputtype='NIFTI_GZ'),
			name='resample'
		)

		extract_signal = Node(
			afni.Maskave(),
			name='extract_signal'
		)

		create_df = Node(
			Function(input_names=['file_list', 'subject_id'],
					output_names=['filename'],
					function=get_timeseries_confounds_file),
			name = 'create_df')

		# DataSink Node - store the wanted results in the wanted repository
		data_sink = Node(
			DataSink(base_directory = self.output_dir),
			name='data_sink',
		)

		data_sink.inputs.substitutions = [('_subject_id_', 'sub-')]

		workflow.connect([
			(info_source, select_files, [('subject_id', 'subject_id')]),
			(select_files, brain_extraction, [('anat', 'in_file')]),
			(brain_extraction, segment, [('out_file', 'in_file')]),
			(segment, extract_maps, [('out_file', 'in_file_a')]),
			(extract_maps, resample, [('out_file', 'in_file')]),
			(select_files, resample, [('func', 'master')]),
			(resample, extract_signal, [('out_file', 'mask')]),
			(select_files, extract_signal, [('func', 'in_file')]),
			(extract_signal, data_sink, [('out_file', 'results_afni.@1D_TS')]),
			(extract_signal, create_df, [('out_file', 'file_list')]),
			(info_source, create_df,[('subject_id', 'subject_id')]),
			(create_df, data_sink, [('filename', 'results_fsl.@df')])
		])

		return workflow