import glob, os
from os.path import join
from nipype.pipeline.engine import Workflow, Node, JoinNode, MapNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import SelectFiles, DataSink
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
from nipype.algorithms.misc import Gunzip
from nilearn import datasets
import nibabel as nib

def get_timeseries_confounds_file(
	file_list: list, 
	subject_id: str
) -> str:
	"""Function to read subjects confounds timeseries and create a dataframe with only white-matter and CSF. 

	Parameters
	----------
	file_list : list
		list of filenames containing subjects confounds 
	subject_id : str
		idx of the subject

	Returns
	-------
	filename : str
		filename of the created dataframe.

	"""
	import pandas as pd
	from os.path import join
	import os 
	
	filename = join(os.getcwd(), f'sub-{subject_id}_task-rest_desc-confounds_timeseries.tsv')
	df_list = [pd.read_csv(f, sep='\s+', header=None, index_col=None) for f in sorted(file_list)]
	
	df = pd.DataFrame({'csf':df_list[0][0].tolist(), 'white_matter': df_list[-1][0].tolist()})
	df.to_csv(filename, header=True, sep='\t', index=False)

	return filename

def get_mean_ts(func, mask_wm, mask_csf):
	'''
	Function to get mean time-serie from a mask. 

	Parameters
	----------
	func : str
		Path to functional file

	mask_wm : Nifti1Image
		Mask to use to extract wm timeseries

	mask_csf : Nifti1Image
		Mask to use to extract csf timeseries

	Returns
	-------
	filename : str
		filename of the created dataframe.
	'''
	from nilearn import image
	from nilearn.regions import img_to_signals_labels
	import pandas as pd
	from os.path import join
	import os 

	mask_wm_res = image.resample_to_img(mask_wm, func, interpolation='nearest')
	mask_csf_res = image.resample_to_img(mask_csf, func, interpolation='nearest')

	mean_ts_wm = img_to_signals_labels(func, mask_wm_res)
	mean_ts_csf = img_to_signals_labels(func, mask_csf_res)
	
	filename = join(os.getcwd(), f'sub-{subject_id}_task-rest_desc-confounds_timeseries.tsv')
	
	df = pd.DataFrame({'csf':mean_ts_csf.tolist(), 'white_matter': mean_ts_wm})
	df.to_csv(filename, header=True, sep='\t', index=False)

	return filename

class Anatomical_Preprocessing:
	'''
	Class to create the anatomical preprocessing pipeline. 

	Attributes
	----------
	subject_list : list of str 
		list of subjects to analyse

	data_dir : str 
		path to data directory

	anat_file_template : str 
		Nipype SelectFiles template for anatomic files

	output_dir : str
		path to output directory

	software : str
		FSL or AFNI 

	pipeline : nipype.Workflow
		Workflow to perform anatomical preprocessing. 
	'''

	def __init__(self, 
		subject_list : list, 
		data_dir : str, 
		func_file_template : str, 
		output_dir : str, 
		anat_file_template : str = '', 
		software : str = 'no-anat'):
		"""
		Parameters
		----------
		subject_list : list of str 
			list of subjects to analyse
		data_dir : str 
			path to data directory
		anat_file_template : str 
			Nipype SelectFiles template for anatomic files
		func_file_template : str
			Nipype SelectFiles template for func files
		output_dir : str
			path to output directory
		software = str
			FSL or AFNI or no-anat
		"""
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
		elif self.software == 'no-anat':
			self.pipeline = self.get_no_anat_wf()

	def get_anat_preproc_fsl_wf(self):
		'''
		Function to create Nipype workflow for extraction of WM and CSF time series. 

		Returns 
		-------
		nipype.Workflow
			Workflow to perform anatomical preprocessing with FSL software package functions. 
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
			(extract_signal, data_sink, [('out_file', 'anat_preproc_fsl.@1D_TS')]),
			(extract_signal, create_df, [('out_file', 'file_list')]),
			(info_source, create_df, [('subject_id', 'subject_id')]),
			(create_df, data_sink, [('filename', 'anat_preproc_fsl.@df')])
		])

		return workflow


	def get_anat_preproc_afni_wf(self):
		'''
		Function to create Nipype workflow for extraction of WM and CSF time series. 

		Returns 
		-------
		nipype.Workflow
			Workflow to perform anatomical preprocessing with AFNI software package functions. 
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

		create_df = JoinNode(
			Function(input_names=['file_list', 'subject_id'],
					output_names=['filename'],
					function=get_timeseries_confounds_file),
            joinsource='extract_maps',
            joinfield='file_list',
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
			(extract_signal, data_sink, [('out_file', 'anat_preproc_afni.@1D_TS')]),
			(extract_signal, create_df, [('out_file', 'file_list')]),
			(info_source, create_df,[('subject_id', 'subject_id')]),
			(create_df, data_sink, [('filename', 'anat_preproc_afni.@df')])
		])

		return workflow

	def get_no_anat_wf(self):

		'''
		Function to create Nipype workflow for extraction of WM and CSF time series. 

		Returns 
		-------
		nipype.Workflow
			Workflow to perform extraction of WM and CSF time series without anatomical priors. 
		'''
		workflow = Workflow('no_anat_wf', base_dir=os.path.join(self.output_dir, 'working_dir'))

		info_source = Node(IdentityInterface(fields=['subject_id']), name='info_source')
		info_source.iterables = [('subject_id', self.subject_list)]

		# Templates to select files node
		file_templates = {
			'func': self.func_file_template
		}
		
		# SelectFiles node - to select necessary files
		select_files = Node(
			SelectFiles(file_templates, base_directory = self.data_dir),
			name='select_files'
		)

		extract_timeseries = Node(Function(
			function = get_mean_ts, 
			input_names = ['func', 'mask_wm', 'mask_csf'],
			output_names = ['filename']),
			name = 'extract_timeseries'
		)

		extract_timeseries.inputs.mask_wm = datasets.load_mni152_wm_mask(resolution = 2)

		extract_timeseries.inputs.mask_csf = nib.load(fsl.Info.standard_image('MNI152_T1_2mm_VentricleMask.nii.gz'))

		# DataSink Node - store the wanted results in the wanted repository
		data_sink = Node(
			DataSink(base_directory = self.output_dir),
			name='data_sink',
		)

		data_sink.inputs.substitutions = [('_subject_id_', 'sub-')]

		workflow.connect([
			(info_source, select_files, [('subject_id', 'subject_id')]),
			(select_files, extract_timeseries, [('func', 'func')]),
			(extract_timeseries, data_sink, [('filename', 'no_anat_preproc.@df')])
		])

		return workflow