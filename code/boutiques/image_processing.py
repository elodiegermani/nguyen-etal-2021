import glob, re, argparse, sys, os
#sys.path.insert(0, '/home/nguyen-etal-2021/nguyenetal/workflows')
from nguyenetal.workflows import anat_preprocessing_wf, confound_reg_wf, func_preprocessing_wf, static_measures_wf 
from os.path import join 

def run_pipeline(
	pipeline: str,
	step: str,
	base_dir: str,
	subject_list: list
	) -> None:
	
	"""
	Create and run the pipeline. 
	"""
	if step == 'func':
		data_dir = f'{base_dir}/.cache/inputs/data/Nifti'
		output_dir = f'{base_dir}/.cache/outputs/{pipeline}'

		preprocessing_wf = func_preprocessing_wf.Functional_Preprocessing(data_dir, 
																		output_dir, 
																		subject_list)
		preprocessing_wf.pipeline.run()

	elif step == 'ica':
		if not os.path.isdir(f'{base_dir}/outputs/{pipeline}/ica-aroma'):
			print('Directory created.')
			os.mkdir(f'{base_dir}/outputs/{pipeline}/ica-aroma')

		for s in subject_list:
			command = f'python3 /home/ICA-AROMA/ICA_AROMA.py \
			-in {base_dir}/outputs/{pipeline}/func_preproc/sub-{s}/sub-{s}_task-rest_bold_mcf_masked_trans.nii.gz \
			-mc {base_dir}/outputs/{pipeline}/func_preproc/sub-{s}/sub-{s}_task-rest_bold_mcf.nii.gz.par \
			-out {base_dir}/outputs/{pipeline}/ica-aroma/sub-{s}'

			os.system(command)

	elif step == 'anat':
		data_dir = f'{base_dir}' 
		output_dir=f'{base_dir}/.cache/outputs/{pipeline}'

		anat_file_template = join('.cache', 'inputs', 'data', 'Nifti', 'sub-{subject_id}', 'anat', 'sub-{subject_id}_T1w.nii.gz')
		func_file_template = join('.cache', 'outputs', f'{pipeline}', 'func_preproc', 'sub-{subject_id}',
								  'sub-{subject_id}_task-rest_bold_mcf_masked_trans.nii.gz')

		if pipeline == 'reproduction_pipeline-fsl_seg':
			software='fsl'
		elif pipeline == 'reproduction_pipeline-afni_seg':
			software = 'afni'
		else:
			software='no-anat'
			anat_file_template = ''

		anat_wf = anat_preprocessing_wf.Anatomical_Preprocessing(subject_list = subject_list, 
															data_dir = data_dir, 
															func_file_template = func_file_template, 
															anat_file_template = anat_file_template,
															output_dir = output_dir, 
															software = software)

		anat_wf.pipeline.run()

	elif step == 'confound':
		data_dir=f'{base_dir}/outputs/{pipeline}'
		output_dir=f'{base_dir}/outputs/{pipeline}'

		if pipeline == 'reproduction_pipeline-fsl_seg':
			software='fsl'
		elif pipeline == 'reproduction_pipeline-afni_seg':
			software = 'afni'
		else:
			software='no-anat'

		wm_csf_template= join(f'anat_preproc_{software}', 'sub-{subject_id}', 
										  'sub-{subject_id}_task-rest_desc-confounds_timeseries.tsv'
										 )

		motion_regressors_template = join('func_preproc','sub-{subject_id}', 
										  'sub-{subject_id}_task-rest_bold_mcf.nii.gz.par'
										 )

		func_file_template = join('func_preproc', 'sub-{subject_id}',
								  'sub-{subject_id}_task-rest_bold_mcf_masked_trans.nii.gz'
								 )

		include_ICA=True
		ica_directory = f'{base_dir}/outputs/{pipeline}/ica-aroma'

		wf_confound = confound_reg_wf.NoiseRegression_Pipeline(subject_list, 
															data_dir, 
															output_dir, 
															wm_csf_template, 
															motion_regressors_template, 
															func_file_template,
															include_ICA, 
															ica_directory)

		wf_confound.pipeline.run()

	elif step == 'feature':
		data_dir = f'{base_dir}/outputs/{pipeline}'
		output_dir = f'{base_dir}/outputs/{pipeline}'

		func_file_template = join('denoising' ,'sub-{subject_id}',
					'sub-{subject_id}_task-rest_bold_mcf_masked_trans_regfilt.nii.gz')
		mask_file_template = join('func_preproc', 'sub-{subject_id}',
				    'sub-{subject_id}_task-rest_bold_mcf_masked_trans_mask.nii.gz')

		cereb_atlas = f'{base_dir}/inputs/atlases/Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz'
		striatum_atlas = f'{base_dir}/inputs/atlases/striatum-con-label-thr25-7sub-2mm.nii.gz'

		alff_reho_wf = static_measures_wf.StaticMeasures_Pipeline(subject_list, 
															data_dir, 
															output_dir, 
															mask_file_template, 
															func_file_template,
															cereb_atlas, 
															striatum_atlas,
															high_pass_filter=0.01, 
															low_pass_filter=0.1, 
															cluster_size=27)

		alff_reho_wf.pipeline.run()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--pipeline', '-p', 
		type=str, 
		choices = [
		'reproduction_pipeline-afni_seg', 
		'reproduction_pipeline-fsl_seg', 
		'reproduction_pipeline-no_anat',
		], 
		help='Pre-processing pipeline')

	parser.add_argument('--step', '-s', type=str, 
		choices = [
		'func',
		'anat', 
		'confound', 
		'feature', 
		'ica'
		], 
		help='Which step to run')

	parser.add_argument('--base_dir', '-b', type=str, help='Absolute path to this directory')
	parser.add_argument('--subject_list', '-l', type=str, help='Subject list')

	args = parser.parse_args()

	subject_list = [int(item) for item in args.subject_list.split(',')]

	print(args.pipeline, args.step, args.base_dir, subject_list)
	run_pipeline(args.pipeline, args.step, args.base_dir, subject_list)


