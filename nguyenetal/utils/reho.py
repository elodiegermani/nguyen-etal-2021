# coding: utf-8
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
#from .reho_utils import compute_reho

def compute_reho(in_file, mask_file, cluster_size):

    """
    Computes the ReHo Map, by computing tied ranks of the timepoints,
    followed by computing Kendall's coefficient concordance(KCC) of a
    timeseries with its neighbours

    Parameters
    ----------

    in_file : nifti file
        4D EPI File

    mask_file : nifti file
        Mask of the EPI File(Only Compute ReHo of voxels in the mask)

    cluster_size : integer
        for a brain voxel the number of neighbouring brain voxels to use for
        KCC.


    Returns
    -------

    out_file : nifti file
        ReHo map of the input EPI image

    """
    import os
    import sys
    import nibabel as nb
    import numpy as np

    def getOpString(mean, std_dev):

        """
        Generate the Operand String to be used in workflow nodes to supply
        mean and std deviation to alff workflow nodes

        Parameters
        ----------

        mean : string
            mean value in string format

        std_dev : string
            std deviation value in string format


        Returns
        -------

        op_string : string


        """

        str1 = "-sub %f -div %f" % (float(mean), float(std_dev))

        op_string = str1 + " -mas %s"

        return op_string


    def f_kendall(timeseries_matrix):

        """
        Calculates the Kendall's coefficient of concordance for a number of
        time-series in the input matrix

        Parameters
        ----------

        timeseries_matrix : ndarray
            A matrix of ranks of a subset subject's brain voxels

        Returns
        -------

        kcc : float
            Kendall's coefficient of concordance on the given input matrix

        """

        import numpy as np
        nk = timeseries_matrix.shape

        n = nk[0]
        k = nk[1]

        sr = np.sum(timeseries_matrix, 1)
        sr_bar = np.mean(sr)

        s = np.sum(np.power(sr, 2)) - n*np.power(sr_bar, 2)

        kcc = 12 *s/np.power(k, 2)/(np.power(n, 3) - n)

        return kcc


    out_file = None

    res_fname = (in_file)
    res_mask_fname = (mask_file)
    CUTNUMBER = 10

    if not (cluster_size == 27 or cluster_size == 19 or cluster_size == 7):
        cluster_size = 27

    nvoxel = cluster_size

    res_img = nb.load(res_fname)
    res_mask_img = nb.load(res_mask_fname)

    res_data = res_img.get_fdata()
    res_mask_data = res_mask_img.get_fdata()

    print(res_data.shape)
    (n_x, n_y, n_z, n_t) = res_data.shape

    # "flatten" each volume of the timeseries into one big array instead of
    # x,y,z - produces (timepoints, N voxels) shaped data array
    res_data = np.reshape(res_data, (n_x*n_y*n_z, n_t), order='F').T

    # create a blank array of zeroes of size n_voxels, one for each time point
    Ranks_res_data = np.tile((np.zeros((1, (res_data.shape)[1]))),
                             [(res_data.shape)[0], 1])

    # divide the number of total voxels by the cutnumber (set to 10)
    # ex. end up with a number in the thousands if there are tens of thousands
    # of voxels
    segment_length = np.ceil(float((res_data.shape)[1])/float(CUTNUMBER))

    for icut in range(0, CUTNUMBER):

        segment = None

        # create a Numpy array of evenly spaced values from the segment
        # starting point up until the segment_length integer
        if not (icut == (CUTNUMBER - 1)):
            segment = np.array(np.arange(icut * segment_length,
                                         (icut+1) * segment_length))
        else:
            segment = np.array(np.arange(icut * segment_length,
                                         (res_data.shape[1])))

        segment = np.int64(segment[np.newaxis])

        # res_data_piece is a chunk of the original timeseries in_file, but
        # aligned with the current segment index spacing
        res_data_piece = res_data[:, segment[0]]
        nvoxels_piece = res_data_piece.shape[1]

        # run a merge sort across the time axis, re-ordering the flattened
        # volume voxel arrays
        res_data_sorted = np.sort(res_data_piece, 0, kind='mergesort')
        sort_index = np.argsort(res_data_piece, axis=0, kind='mergesort')

        # subtract each volume from each other
        db = np.diff(res_data_sorted, 1, 0)

        # convert any zero voxels into "True" flag
        db = db == 0

        # return an n_voxel (n voxels within the current segment) sized array
        # of values, each value being the sum total of TRUE values in "db"
        sumdb = np.sum(db, 0)

        temp_array = np.array(np.arange(0, n_t))
        temp_array = temp_array[:, np.newaxis]

        sorted_ranks = np.tile(temp_array, [1, nvoxels_piece])

        if np.any(sumdb[:]):

            tie_adjust_index = np.flatnonzero(sumdb)

            for i in range(0, len(tie_adjust_index)):

                ranks = sorted_ranks[:, tie_adjust_index[i]]

                ties = db[:, tie_adjust_index[i]]

                tieloc = np.append(np.flatnonzero(ties), n_t + 2)
                maxties = len(tieloc)
                tiecount = 0

                while(tiecount < maxties -1):
                    tiestart = tieloc[tiecount]
                    ntied = 2
                    while(tieloc[tiecount + 1] == (tieloc[tiecount] + 1)):
                        tiecount += 1
                        ntied += 1

                    ranks[tiestart:tiestart + ntied] = np.ceil(np.float32(np.sum(ranks[tiestart:tiestart + ntied ]))/np.float32(ntied))
                    tiecount += 1

                sorted_ranks[:, tie_adjust_index[i]] = ranks

        del db, sumdb
        sort_index_base = np.tile(np.multiply(np.arange(0, nvoxels_piece), n_t), [n_t, 1])
        sort_index += sort_index_base
        del sort_index_base

        ranks_piece = np.zeros((n_t, nvoxels_piece))

        ranks_piece = ranks_piece.flatten(order='F')
        sort_index = sort_index.flatten(order='F')
        sorted_ranks = sorted_ranks.flatten(order='F')

        ranks_piece[sort_index] = np.array(sorted_ranks)

        ranks_piece = np.reshape(ranks_piece, (n_t, nvoxels_piece), order='F')

        del sort_index, sorted_ranks

        Ranks_res_data[:, segment[0]] = ranks_piece

        sys.stdout.write('.')

    Ranks_res_data = np.reshape(Ranks_res_data, (n_t, n_x, n_y, n_z), order='F')

    K = np.zeros((n_x, n_y, n_z))

    mask_cluster = np.ones((3, 3, 3))

    if nvoxel == 19:
        mask_cluster[0, 0, 0] = 0
        mask_cluster[0, 2, 0] = 0
        mask_cluster[2, 0, 0] = 0
        mask_cluster[2, 2, 0] = 0
        mask_cluster[0, 0, 2] = 0
        mask_cluster[0, 2, 2] = 0
        mask_cluster[2, 0, 2] = 0
        mask_cluster[2, 2, 2] = 0

    elif nvoxel == 7:

        mask_cluster[0, 0, 0] = 0
        mask_cluster[0, 1, 0] = 0
        mask_cluster[0, 2, 0] = 0
        mask_cluster[0, 0, 1] = 0
        mask_cluster[0, 2, 1] = 0
        mask_cluster[0, 0, 2] = 0
        mask_cluster[0, 1, 2] = 0
        mask_cluster[0, 2, 2] = 0
        mask_cluster[1, 0, 0] = 0
        mask_cluster[1, 2, 0] = 0
        mask_cluster[1, 0, 2] = 0
        mask_cluster[1, 2, 2] = 0
        mask_cluster[2, 0, 0] = 0
        mask_cluster[2, 1, 0] = 0
        mask_cluster[2, 2, 0] = 0
        mask_cluster[2, 0, 1] = 0
        mask_cluster[2, 2, 1] = 0
        mask_cluster[2, 0, 2] = 0
        mask_cluster[2, 1, 2] = 0
        mask_cluster[2, 2, 2] = 0

    for i in range(1, n_x - 1):
        for j in range(1, n_y -1):
            for k in range(1, n_z -1):

                block = Ranks_res_data[:, i-1:i+2, j-1:j+2, k-1:k+2]
                mask_block = res_mask_data[i-1:i+2, j-1:j+2, k-1:k+2]

                if not(int(mask_block[1, 1, 1]) == 0):

                    if nvoxel == 19 or nvoxel == 7:
                        mask_block = np.multiply(mask_block, mask_cluster)

                    R_block = np.reshape(block, (block.shape[0], 27),
                                         order='F')
                    mask_R_block = R_block[:, np.argwhere(np.reshape(mask_block, (1, 27), order='F') > 0)[:, 1]]

                    K[i, j, k] = f_kendall(mask_R_block)

    img = nb.Nifti1Image(K, header=res_img.header,
                         affine=res_img.affine)
    reho_file = os.path.join(os.getcwd(), 'ReHo.nii.gz')
    img.to_filename(reho_file)
    out_file = reho_file

    return out_file



def create_reho(wf_name):

    """
    Regional Homogeneity(ReHo) approach to fMRI data analysis

    This workflow computes the ReHo map, z-score on map

    Parameters
    ----------

    None

    Returns
    -------
    reHo : workflow
        Regional Homogeneity Workflow

    Notes
    -----

    `Source <https://github.com/FCP-INDI/C-PAC/blob/master/CPAC/reho/reho.py>`_

    Workflow Inputs: ::

        inputspec.rest_res_filt : string (existing nifti file)
            Input EPI 4D Volume

        inputspec.rest_mask : string (existing nifti file)
            Input Whole Brain Mask of EPI 4D Volume

        inputspec.cluster_size : integer
            For a brain voxel the number of neighbouring brain voxels to use for KCC.
            Possible values are 27, 19, 7. Recommended value 27


    Workflow Outputs: ::

        outputspec.raw_reho_map : string (nifti file)

        outputspec.z_score : string (nifti file)


    ReHo Workflow Procedure:

    1. Generate ReHo map from the input EPI 4D volume, EPI mask and cluster_size
    2. Compute Z score of the ReHo map by subtracting mean and dividing by standard deviation

    .. exec::
        from CPAC.reho import create_reho
        wf = create_reho()
        wf.write_graph(
            graph2use='orig',
            dotfilename='./images/generated/reho.dot'
        )

    Workflow Graph:

    .. image:: ../../images/generated/reho.png
        :width: 500

    Detailed Workflow Graph:

    .. image:: ../../images/generated/reho_detailed.png
        :width: 500

    References
    ----------
    .. [1] Zang, Y., Jiang, T., Lu, Y., He, Y.,  Tian, L. (2004). Regional homogeneity approach to fMRI data analysis. NeuroImage, 22(1), 394, 400. doi:10.1016/j.neuroimage.2003.12.030

    Examples
    --------
    >>> from CPAC import reho
    >>> wf = reho.create_reho('reho')
    >>> wf.inputs.inputspec.rest_res_filt = '/home/data/Project/subject/func/rest_res_filt.nii.gz'  # doctest: +SKIP
    >>> wf.inputs.inputspec.rest_mask = '/home/data/Project/subject/func/rest_mask.nii.gz'  # doctest: +SKIP
    >>> wf.inputs.inputspec.cluster_size = 27
    >>> wf.run()  # doctest: +SKIP
    """

    reHo = pe.Workflow(name=wf_name)
    inputNode = pe.Node(util.IdentityInterface(fields=['cluster_size',
                                                       'rest_res_filt',
                                                       'rest_mask']),
                        name='inputspec')

    outputNode = pe.Node(util.IdentityInterface(fields=['raw_reho_map']),
                         name='outputspec')


    raw_reho_map = pe.Node(util.Function(input_names=['in_file', 'mask_file',
                                                      'cluster_size'],
                                         output_names=['out_file'],
                                         function=compute_reho),
                           name='reho_map', mem_gb=6.0)

    reHo.connect(inputNode, 'rest_res_filt', raw_reho_map, 'in_file')
    reHo.connect(inputNode, 'rest_mask', raw_reho_map, 'mask_file')
    reHo.connect(inputNode, 'cluster_size', raw_reho_map, 'cluster_size')
    reHo.connect(raw_reho_map, 'out_file', outputNode, 'raw_reho_map')

    return reHo


def reho(wf, cfg, strat_pool, pipe_num, opt=None):
    '''
    {"name": "ReHo",
     "config": ["regional_homogeneity"],
     "switch": ["run"],
     "option_key": "None",
     "option_val": "None",
     "inputs": ["desc-preproc_bold",
                "space-bold_desc-brain_mask"],
     "outputs": ["reho"]}
    '''
    cluster_size = cfg.regional_homogeneity['cluster_size']

    # Check the cluster size is supported
    if cluster_size not in [7, 19, 27]:
        err_msg = 'Cluster size specified: %d, is not ' \
                  'supported. Change to 7, 19, or 27 and try ' \
                  'again' % cluster_size
        raise Exception(err_msg)

    reho = create_reho(f'reho_{pipe_num}')
    reho.inputs.inputspec.cluster_size = cluster_size

    node, out = strat_pool.get_fdata("desc-preproc_bold")
    wf.connect(node, out, reho, 'inputspec.rest_res_filt')

    node, out_file = strat_pool.get_fdata('space-bold_desc-brain_mask')
    wf.connect(node, out_file, reho, 'inputspec.rest_mask')

    outputs = {
        'reho': (reho, 'outputspec.raw_reho_map')
    }

    return (wf, outputs)


def reho_space_template(wf, cfg, strat_pool, pipe_num, opt=None):
    '''
    {"name": "ReHo_space_template",
     "config": ["regional_homogeneity"],
     "switch": ["run"],
     "option_key": "None",
     "option_val": "None",
     "inputs": ["space-template_res-derivative_desc-preproc_bold",
                "space-template_res-derivative_desc-bold_mask"],
     "outputs": ["space-template_reho"]}
    '''
    cluster_size = cfg.regional_homogeneity['cluster_size']

    # Check the cluster size is supported
    if cluster_size not in [7, 19, 27]:
        err_msg = 'Cluster size specified: %d, is not ' \
                  'supported. Change to 7, 19, or 27 and try ' \
                  'again' % cluster_size
        raise Exception(err_msg)

    reho = create_reho(f'reho_{pipe_num}')
    reho.inputs.inputspec.cluster_size = cluster_size

    node, out = strat_pool.get_data("space-template_res-derivative_desc-preproc_bold")
    wf.connect(node, out, reho, 'inputspec.rest_res_filt')

    node, out_file = strat_pool.get_data(
        'space-template_res-derivative_desc-bold_mask')
    wf.connect(node, out_file, reho, 'inputspec.rest_mask')

    outputs = {
        'space-template_reho': (reho, 'outputspec.raw_reho_map')
    }

    return (wf, outputs)
