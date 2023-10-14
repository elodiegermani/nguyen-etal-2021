""" Workflow to perform preprocessing of functional images as part of the reproduction pipeline.
"""

from nipype.interfaces.ants import Registration
from nipype.interfaces.ants import ApplyTransforms
from nipype import Node, Workflow, MapNode
import nipype.interfaces.fsl as fsl
import nipype.interfaces.afni as afni
from nipype.interfaces.robex.preprocess import RobexSegment
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from os.path import join

def select_volume(
	filename : str, 
	which : str
) -> int :
	"""Return the index of a file

	Parameters
	----------

	filename : str
		filename of the nifti image
	which : str
		one of 'first' or 'middle'

	Returns
	-------

	int 
		idx of the "which" volume in the image.
	"""
	from nibabel import load
	import numpy as np
	
	if which.lower() == 'first':
		idx = 0
	elif which.lower() == 'middle':
		idx = int(np.ceil(load(filename).shape[3]/2))
	else:
		raise Exception('unknown value for volume selection : %s'%which)
		
	return idx

class Functional_Preprocessing():
	"""
	Class representing the workflow used to perform preprocessing of the functional images as part of the reproduction pipeline.

	Attributes
	----------
	data_dir : str
		path to the directory containing raw images
	output_dir : str
		path to the directory where to store results
	subject_list : list
		list of subjects for which to analyse the images
	pipeline : nipype.Workflow
		Functional preprocessing workflow.
	"""

	def __init__(self, 
		data_dir : str, 
		output_dir : str, 
		subject_list : list
	):
		"""
		Parameters
		----------

		data_dir : str
			path to the directory containing raw images
		output_dir : str
			path to the directory where to store results
		subject_list : list
			list of subjects for which to analyse the images
		"""

		self.data_dir = data_dir
		self.output_dir = output_dir
		self.subject_list = subject_list

		self.pipeline = self.get_preprocessing_wf()

		self.mask_pipeline = self.get_mask_wf()

	def get_reg_wf(self, reg_type:str='func', name:str='func_reg_wf'): 
		"""
		Return the registration workflow for direct EPI-to-template registration. 

		Parameters
		----------
		reg_type : str
			Type of the registration 
		name : str
			Name of the workflow 

		Returns
		-------
		nipype.Workflow 
			Workflow for EPI-to-template registration

		"""
		# fetch input 
		inputnode = Node(IdentityInterface(fields=['in_file', 'template']),
							name='inputnode')
		
		outputnode = Node(IdentityInterface(fields=['registered_image', 'warped_image', 'transform']), 
							name='outputnode')
		
		extract_ref = Node(interface=fsl.ExtractROI(t_size=1),
						  name = 'extractref')
		
		# registration or normalization step based on symmetric diffeomorphic image registration (SyN) using ANTs 
		reg = Node(Registration(), name='NormalizationAnts')
		reg.inputs.output_transform_prefix = f'{reg_type}2template'
		reg.inputs.output_warped_image = f'{reg_type}2template.nii.gz'
		reg.inputs.output_transform_prefix = f'{reg_type}2template_'
		reg.inputs.transforms = ['Rigid','Affine', 'SyN']
		reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.2, 3.0, 0.0)]
		reg.inputs.number_of_iterations = ([[10000, 111110, 11110]] * 2 + [[40, 10, 5]])
		reg.inputs.dimension = 3
		reg.inputs.write_composite_transform = True
		reg.inputs.collapse_output_transforms = True
		reg.inputs.initial_moving_transform_com = True
		reg.inputs.metric = ['Mattes'] * 2 + [['Mattes', 'CC']]
		reg.inputs.metric_weight = [1] * 2 + [[0.5, 0.5]]
		reg.inputs.radius_or_number_of_bins = [32] * 2 + [[32, 4]]
		reg.inputs.sampling_strategy = ['Regular'] * 2 + [[None, None]]
		reg.inputs.sampling_percentage = [0.3] * 2 + [[None, None]]
		reg.inputs.convergence_threshold = [1.e-8] * 2 + [-0.01]
		reg.inputs.convergence_window_size = [20] * 2 + [5]
		reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 2 + [[1, 0.5, 0]]
		reg.inputs.sigma_units = ['vox'] * 3
		reg.inputs.shrink_factors = [[3, 2, 1]]*2 + [[4, 2, 1]]
		reg.inputs.use_estimate_learning_rate_once = [True] * 3
		reg.inputs.use_histogram_matching = [False] * 2 + [True]
		reg.inputs.winsorize_lower_quantile = 0.005
		reg.inputs.winsorize_upper_quantile = 0.995
		reg.inputs.args = '--float'
		
		# apply the transform 
		apply_trans_func = Node(ApplyTransforms(args='--float',
										input_image_type=3,
										interpolation='BSpline',
										invert_transform_flags=[False],
										num_threads=1,
										terminal_output='file'), 
								name='apply_trans_func')
		
		reg_wf = Workflow(name=name, base_dir= self.output_dir)	

		if reg_type=='func':
			reg_wf.connect(
				[
					(inputnode, extract_ref, [('in_file', 'in_file'),
										  (('in_file', select_volume, 'middle'), 't_min')]),
					(extract_ref, reg, [('roi_file', 'moving_image')]), 
				]
			)
		else: 
			reg_wf.connect(
				[
					(inputnode, reg, [('in_file', 'moving_image')])
				]
			)
		
		reg_wf.connect(
			[
			   (inputnode, reg, [('template', 'fixed_image')]),
			   (inputnode, apply_trans_func, [('template', 'reference_image'),
											   ('in_file', 'input_image')]),
			   (reg, apply_trans_func, [('composite_transform', 'transforms')]),
			   (reg, outputnode, [('warped_image', 'warped_image'),
			   					('composite_transform', 'transform')]),
			   (apply_trans_func, outputnode, [('output_image', 'registered_image')])
			]
		)
		
		return reg_wf 

	def get_preprocessing_wf(self):
		"""
		Function to create the preprocessing workflow.

		Returns
		-------
		nipype.Workflow
			Functional preprocessing workflow.

		"""
		# IdentityInterface node - allows to iterate over subjects and runs
		info_source = Node(
			IdentityInterface(fields=['subject_id']),
			name='info_source'
		)
		info_source.iterables = [
			('subject_id', self.subject_list)
		]
		
		# Templates to select files node 
		file_templates = {
			'anat': join(
				'sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz'
				),
			'func': join(
				'sub-{subject_id}', 'func', 'sub-{subject_id}_task-rest_bold.nii.gz'
				)
		}
		
		# SelectFiles node - to select necessary files
		select_files = Node(
			SelectFiles(file_templates, base_directory = self.data_dir),
			name='select_files'
		)
		
		# DataSink Node - store the wanted results in the wanted repository
		data_sink = Node(
			DataSink(base_directory = self.output_dir),
			name='data_sink',
		)

		data_sink.inputs.substitutions = [('_subject_id_', 'sub-')]

		# Extract a reference volume from 4D image
		extract_ref = Node(
			interface=fsl.ExtractROI(t_size=1),
			name='extractref')

		# Motion correction with McFLIRT
		motion_correction = Node(
			fsl.MCFLIRT(dof=6,
						save_plots=True),
			name='motion_correction'
		)

		# Skullstripping
		skullstrip_func = Node(
			afni.Automask(outputtype='NIFTI_GZ'),
			name='skullstrip_func'
		)
		
		# Registration 
		non_linear_registration_func = self.get_reg_wf('func', 'func_reg_wf')
		non_linear_registration_func.inputs.inputnode.template=fsl.Info.standard_image('MNI152_T1_2mm_brain.nii.gz')
		

		# Compute mask after preprocessing 
		# Skullstripping
		mask_func = Node(
			afni.Automask(outputtype='NIFTI_GZ'),
			name='mask_func'
		)
		# Global workflow 
		preproc_wf = Workflow(name='preproc_wf')
		preproc_wf.base_dir = self.output_dir

		preproc_wf.connect( # Connection between nodes
			[
				(info_source, select_files, [('subject_id', 'subject_id')]),
				(select_files, motion_correction, [('func', 'in_file')]),
				(select_files, extract_ref, [(('func', select_volume, 'middle'), 't_min'),
											 ('func', 'in_file')]),
				(extract_ref, motion_correction, [('roi_file', 'ref_file')]),
				(motion_correction, skullstrip_func, [('out_file', 'in_file')]),
				(skullstrip_func, non_linear_registration_func, [('brain_file', 'inputnode.in_file')]),
				(non_linear_registration_func, mask_func, [('outputnode.registered_image', 'in_file')]),
				(mask_func, data_sink, [('out_file', 'func_preproc.@mask')]),
				(skullstrip_func, data_sink, [('out_file', 'func_preproc.@skullstriped')]),
				(motion_correction, data_sink, [('par_file', 'func_preproc.@motion_param')]),
				(non_linear_registration_func, data_sink, [('outputnode.registered_image', 'func_preproc.@registered_func')]),
				(non_linear_registration_func, data_sink, [('outputnode.warped_image', 'func_preproc.@warped_image'),
															('outputnode.transform', 'func_preproc.@transform')]),
			]
		)

		return preproc_wf