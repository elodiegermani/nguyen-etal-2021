from nguyenetal.prediction import train_model
import glob, re, argparse, sys, os
from os.path import join 

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--pipeline', '-p', type=str, choices = ['reproduction_pipeline-afni_seg', 
		'reproduction_pipeline-fsl_seg', 
		'reproduction_pipeline-no_anat'], 
		help='Pre-processing pipeline')
	parser.add_argument('--specific', '-s', type=str, help='Specificity of the training')
	parser.add_argument('--timepoints', '-t', type=str, 
		help='Timepoints to train')
	parser.add_argument('--features', '-f', type=str, 
		help='Features to train')
	parser.add_argument('--atlases', '-a', type=str, 
		help='Atlases to train')

	args = parser.parse_args()

	timepoints = [str(item) for item in args.timepoints.split(',')]
	features = [str(item) for item in args.features.split(',')]
	atlases = [str(item) for item in args.atlases.split(',')]

	print(args.pipeline, args.specific, timepoints, features, atlases)
	# train_model.train_ml_models(pipeline, 
    #                 specific, 
    #                 timepoints = ['baseline', '1y', '2y', '4y'], 
    #                 features = ['zfalff', 'zReHo'], 
    #                 atlases = ['schaefer', 'basc197', 'basc444'])


