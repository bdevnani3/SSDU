# import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import os
import h5py as h5
import utils
import tf_utils
import parser_ops
import masks.ssdu_masks as ssdu_masks
import UnrollNet
from tensorflow.python.framework import ops
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.INFO)

from fastmri.data.subsample import create_mask_for_mask_type

# if __name__ == "main":
parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

save_dir ='saved_models'
directory = os.path.join(save_dir, 'SSDU_' + args.data_opt + '_' +str(args.epochs)+'Epochs_Rate'+ str(args.acc_rate) +\
                         '_' + str(args.nb_unroll_blocks) + 'Unrolls_' + args.mask_type+'Selection' )

if not os.path.exists(directory):
    os.makedirs(directory)

# tf.logging.info('\n create a test model for the testing')
# test_graph_generator = tf_utils.test_graph(directory)

#...........................................................................d....
start_time = time.time()
tf.logging.info('.................SSDU Training.....................')
ops.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# .......................Load the Data..........................................
tf.logging.info('\n Loading {args.data_opt} data, acc rate : {args.acc_rate} mask type : {args.mask_type}')

##TODO(): Use all the training data, not just one file
kspace_dir = '/srv/share4/ksarangmath3/mri/data/singlecoil_train/file1000017.h5'
kspace_train = h5.File(kspace_dir, "r")['kspace'][:]

## Adding dimension for coil
kspace_train = np.expand_dims(kspace_train,3)
kspace_shape = kspace_train.shape

##TODO(): GET actual sensitivity maps 
sens_maps = np.ones(kspace_shape)

tf.logging.info('\n Normalize the kspace to 0-1 region')
for ii in range(np.shape(kspace_train)[0]):
    kspace_train[ii, :, :, :] = kspace_train[ii, :, :, :] / np.max(np.abs(kspace_train[ii, :, :, :][:]))

nSlices, *_ = kspace_train.shape

#TODO(): Use generated masks
# mask_fn = create_mask_for_mask_type("random",[0.08], [4])
# original_mask = mask_fn((1,34,640,372), 1)
original_mask = np.zeros((args.nrow_GLOB, args.ncol_GLOB))
original_mask[:,:50] = 1

tf.logging.info(f'\n size of kspace: {kspace_train.shape}, maps: {sens_maps.shape}, mask: {original_mask.shape}')


trn_mask, loss_mask = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                      np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

nw_input = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
ref_kspace = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)

tf.logging.info('\n create training and loss masks and generate network inputs... ')
ssdu_masker = ssdu_masks.ssdu_masks()
for ii in range(nSlices):
    if np.mod(ii, 50) == 0:
        tf.logging.info(f'\n Iteration: {ii}')

    if args.mask_type == 'Gaussian':
        # tf.logging.info(original_mask)
        trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.Gaussian_selection(kspace_train[ii], original_mask, num_iter=ii)

    elif args.mask_type == 'Uniform':
        trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.uniform_selection(kspace_train[ii], original_mask, num_iter=ii)

    else:
        raise ValueError('Invalid mask selection')

    sub_kspace = kspace_train[ii] * np.tile(trn_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    ref_kspace[ii, ...] = kspace_train[ii] * np.tile(loss_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    nw_input[ii, ...] = utils.sense1(sub_kspace, sens_maps[ii, ...])

# %%  zeropadded outer edges of k-space with no signal- check github readme file for explanation for further explanations
# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal
# in the training mask we set corresponding columns as 1 to ensure data consistency

##TODO(): Add similar block for our dataset
if args.data_opt == 'Coronal_PD':
    trn_mask[:, :, 0:17] = np.ones((nSlices, args.nrow_GLOB, 17))
    trn_mask[:, :, 352:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 16))

# %% Prepare the data for the training
sens_maps = np.transpose(sens_maps, (0, 3, 1, 2))
ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))
nw_input = utils.complex2real(nw_input)

tf.logging.info(f'\n size of ref kspace: , {ref_kspace.shape}, nw_input:  {nw_input.shape},  maps: {sens_maps.shape}, mask: {trn_mask.shape}')

# %% set the batch size
total_batch = int(np.floor(np.float32(nw_input.shape[0]) / (args.batchSize)))
kspaceP = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='refkspace')
sens_mapsP = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='sens_maps')
trn_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='trn_mask')
loss_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='loss_mask')
nw_inputP = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, 2), name='nw_input')

# %% creating the dataset
dataset = tf.data.Dataset.from_tensor_slices((kspaceP, nw_inputP, sens_mapsP, trn_maskP, loss_maskP))
dataset = dataset.shuffle(buffer_size=10 * args.batchSize)
dataset = dataset.batch(args.batchSize)
dataset = dataset.prefetch(args.batchSize)
iterator = dataset.make_initializable_iterator()
ref_kspace_tensor, nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor = iterator.get_next('getNext')

# %% make training model
nw_output_img, nw_output_kspace, *_ = UnrollNet.UnrolledNet(nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor).model
scalar = tf.constant(0.5, dtype=tf.float32)
loss = tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace) / tf.norm(ref_kspace_tensor)) + \
       tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(ref_kspace_tensor, ord=1))

all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

saver = tf.train.Saver(max_to_keep=100)
sess_trn_filename = os.path.join(directory, 'model')
totalLoss = []
avg_cost = 0
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    tf.logging.info(f'SSDU Parameters: Epochs: {args.epochs} Batch Size: {args.batchSize}, Number of trainable parameters: {sess.run(all_trainable_vars)}, Iterations: {total_batch}')
    feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: trn_mask, loss_maskP: loss_mask, sens_mapsP: sens_maps}

    tf.logging.info('Training...')
    for ep in range(1, args.epochs + 1):
        sess.run(iterator.initializer, feed_dict=feedDict)
        avg_cost = 0
        tic = time.time()
        try:
            for jj in range(total_batch):
                tmp, _, _ = sess.run([loss, update_ops, optimizer])
                avg_cost += tmp / total_batch
            toc = time.time() - tic
            totalLoss.append(avg_cost)
            tf.logging.info(f'Epoch: {ep} elapsed_time =""{toc}", "cost =", "{avg_cost}"')

        except tf.errors.OutOfRangeError:
            pass

        if (np.mod(ep, 10) == 0):
            saver.save(sess, sess_trn_filename, global_step=ep)
            sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})

end_time = time.time()
sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})
tf.logging.info('Training completed in  ', ((end_time - start_time) / 60), ' minutes')
