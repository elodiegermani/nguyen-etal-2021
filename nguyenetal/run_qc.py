from nilearn import image, plotting, masking
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image as ImageDisplay
import pandas as pd 
import seaborn as sns
import glob
import os, sys
import numpy as np 
import pandas as pd 

def make_gif(frame_folder, output_name, sub):
	frames = [Image.open(image) for image in [f"{frame_folder}/registration_sub-{sub}_index-{f}.png" for f in range(210)]]
	frame_one = frames[0]
	frame_one.save(f'{frame_folder}/{output_name}', format="GIF", append_images=frames,
			   save_all=True, duration=100, loop=0)

def make_gif_anat(frame_folder, output_name, subject_list, seg='wm'):
	frames = [Image.open(image) for image in [f"{frame_folder}/{seg}_seg_sub-{sub}.png" for sub in subject_list]]
	frame_one = frames[0]
	frame_one.save(f'{frame_folder}/{output_name}', format="GIF", append_images=frames,
			   save_all=True, duration=50, loop=0)

# load the datafile
def compute_fd(motpars):

	# compute absolute displacement
	dmotpars=np.zeros(motpars.shape)
	
	dmotpars[1:,:]=np.abs(motpars[1:,:] - motpars[:-1,:])
	
	# convert rotation to displacement on a 50 mm sphere
	# mcflirt returns rotation in radians
	# from Jonathan Power:
	#The conversion is simple - you just want the length of an arc that a rotational
	# displacement causes at some radius. Circumference is pi*diameter, and we used a 5
	# 0 mm radius. Multiply that circumference by (degrees/360) or (radians/2*pi) to get the 
	# length of the arc produced by a rotation.
	
	
	headradius=50
	disp=dmotpars.copy()
	disp[:,0:3]=np.pi*headradius*2*(disp[:,0:3]/(2*np.pi))
	
	FD=np.sum(disp,1)
	
	return FD

def main(base_dir):
	subject_list = sorted([f.split('/')[-1].split('-')[-1] \
							for f in glob.glob(f'{base_dir}/inputs/data/Nifti/sub-*')])

	n_comp_list = []

	n_outliers = 0

	for sub in subject_list:
		print('QC subject', sub)
		img = nib.load(f'{out_dirs}/func_preproc/sub-{sub}/sub-{sub}_task-rest_bold_mcf_masked_trans.nii.gz')
		# Registration check
		if not os.path.exists(f'{base_dir}/figures/qc/func/registration_sub-{sub}_index-210.png'):
			for i in range(210):
				plotting.plot_roi('/opt/fsl-6.0.6.1/data/standard/MNI152_T1_2mm_brain_mask.nii.gz', image.index_img(img,i),
								 cut_coords=(0, -20, 10), view_type='contours', title = f'Timepoint={i}')
				plt.savefig(f'{base_dir}/figures/qc/func/registration_sub-{sub}_index-{i}.png')
				plt.close()

		qc_dir = f'{base_dir}/figures/qc/func'
		animation_file = f'sub-{sub}_reg-qc.gif'
		make_gif(qc_dir, output_name=animation_file, sub=sub)
		print('Registration: DONE.')

		out_dirs = f'{base_dir}/outputs/reproduction_pipeline-fsl_seg'
		anat = f'{base_dir}/inputs/data/Nifti/sub-{sub}/anat/sub-{sub}_T1w.nii.gz'
		img_wm = f'{out_dirs}/anat_preproc_fsl/sub-{sub}/sub-{sub}_T1w_brain_seg_2.nii.gz'

		plotting.plot_roi(img_wm, display_mode='ortho', bg_img = anat, 
							view_type='contours',
							cut_coords=(10, -20, 10))
		plt.savefig(f'{base_dir}/figures/qc/seg/wm_seg-fsl_sub-{sub}.png')
		plt.close()

		img_csf = f'{out_dirs}/anat_preproc_fsl/sub-{sub}/sub-{sub}_T1w_brain_seg_0.nii.gz'

		plotting.plot_roi(img_csf, display_mode='ortho', bg_img = anat, 
							view_type='contours',
							cut_coords=(10, -20, 10))
		plt.savefig(f'{base_dir}/figures/qc/seg/csf_seg-fsl_sub-{sub}.png')
		plt.close()

		out_dirs = f'{base_dir}/outputs/reproduction_pipeline-afni_seg'
		f = f'{out_dirs}/anat_preproc_afni/sub-{sub}/Classes+orig.BRIK'
		im = nib.load(f)
		im_nii = nib.Nifti1Image(im.get_fdata(), im.affine)

		plotting.plot_roi(im_nii, display_mode='ortho', bg_img = anat, 
							view_type='contours',
							cut_coords=(10, -20, 10))
		plt.savefig(f'{base_dir}/figures/qc/seg/seg-afni_sub-{sub}.png')
		plt.close()
		print('Segmentation: DONE.')

		df_mp = pd.read_csv(f'{out_dirs}/func_preproc/sub-{sub}/sub-{sub}_task-rest_bold_mcf.nii.gz.par', sep='\s\s', header=None)
		df_mp.columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
		for i in range(3):
			sns.lineplot(x=range(210), y=df_mp[df_mp.columns[i]], label = df_mp.columns[i])
		plt.legend()
		plt.savefig(f'{base_dir}/figures/qc/func/motion-param-trans_sub-{sub}.png')
		plt.close()

		for i in range(3,6):
			sns.lineplot(x=range(210), y=df_mp[df_mp.columns[i]], label = df_mp.columns[i])
		plt.legend()
		plt.savefig(f'{base_dir}/figures/qc/func/motion-param-rot_sub-{sub}.png')
		plt.close()

		fd = compute_fd(np.array(df_mp))

		if (np.any(fd>0.55)):
			n_outliers += 1

		sns.lineplot(x=range(210), y=fd, label = 'Framewise displacement')
		plt.savefig(f'{base_dir}/figures/qc/func/fd_sub-{sub}.png')
		plt.close()
		print('Motion regressors: DONE.')

		ica_comp = f'{out_dirs}/ica-aroma/sub-{sub}/classified_motion_ICs.txt'

		with open(ica_comp, 'r') as f:
			ica_comp_list = f.read().split(',')

		print('Subject', sub, ', number of motion-related components:', len(ica_comp_list))

		n_comp_list.append(len(ica_comp_list))
		print('ICA: DONE.')

		df_confounds = pd.read_csv(f'{out_dirs}/denoising/sub-{sub}/sub-{sub}_task-rest_desc-confounds_timeseries.tsv', 
			sep='\t', header=None)
		mask_img = nib.load(f'{out_dirs}/func/sub-{sub}/sub-{sub}_task-rest_bold_mcf_masked_trans_mask.nii.gz')
		masked_ts = masking.apply_mask(img, mask_img)

		global_signal = []
		for i in range(210):
			global_signal.append(np.mean(masked_ts[i]))

		correlations = []

		for cols in df_confounds.columns:
			correlations.append(np.corrcoef(global_signal, df_confounds[cols].tolist())[0,1])

		sns.barplot(x=df_confounds.columns, y=correlations)
		plt.xticks(rotation=70)
		plt.tight_layout()
		plt.savefig(f'{base_dir}/figures/qc/func/confounds-correlation_sub-{sub}.png')
		plt.close()
		print('CONFOUNDS: DONE.')


	print('N. of outlier subjects:', n_outliers)

	sns.histplot(x=n_comp_list)
	plt.savefig(f'{base_dir}/figures/qc/ica/hist_motion-related-comp.png')
	plt.close()

if __name__ == '__main__':
	base_dir = '/nfs/nas-empenn/data/share/users/egermani/nguyen-etal-2021'
	main(base_dir)
	