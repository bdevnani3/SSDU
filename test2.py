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
import matplotlib.pyplot as plt

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data import transforms
from fastmri.data import subsample

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


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
tf.logging.info(f' Loading {args.data_opt} data, acc rate : {args.acc_rate} mask type : {args.mask_type}')

def center_crop(data, shape, row_offset=0):
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")
    if row_offset == 0:
        w_from = (data.shape[-2] - shape[0]) // 2
        h_from = (data.shape[-1] - shape[1]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        return data[..., w_from:w_to, h_from:h_to].copy()
    if row_offset == -1:
        w_from = 0
        h_from = (data.shape[-1] - shape[1]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        return data[..., :w_to, h_from:h_to].copy()
    if row_offset == 1:
        w_from = data.shape[-2] - shape[0]
        h_from = (data.shape[-1] - shape[1]) // 2
        # w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        print(data[..., w_from:, h_from:h_to].shape)
        print(w_from)
        return data[..., w_from:, h_from:h_to].copy()

all_crops = False

try:
    # Saving as this compressed np array saves pre-processing time
    print("Trying to use pre-loaded data")
    kspace_train = np.load('data/kspace_test_center_crops_small.npz')["kspace_train"]
except:
    kspace_train = None
    train_directory = "/srv/share4/ksarangmath3/mri/small_data/singlecoil_test/"
    print("Generating Dataset")
    for i,filename in enumerate(os.listdir(train_directory)):
        print(i)
        full_file_path = train_directory + filename
        if kspace_train is None:
            kspace_train = h5.File(full_file_path, "r")['kspace'][:]
            if not all_crops:
                kspace_train = center_crop(kspace_train, (320,320),0)
                kspace_train = np.expand_dims(kspace_train,3)
                kspace_train = [kspace_train]
            else:
                kspace_train0 = center_crop(kspace_train, (320,320),0)
                kspace_train1 = center_crop(kspace_train, (320,320),-1)
                kspace_train2 = center_crop(kspace_train, (320,320),1)
                kspace_train0 = np.expand_dims(kspace_train0,3)
                kspace_train1 = np.expand_dims(kspace_train1,3)
                kspace_train2 = np.expand_dims(kspace_train2,3)
                kspace_train = [kspace_train0,kspace_train1,kspace_train2]
        else:
            temp = h5.File(full_file_path, "r")['kspace'][:]
            if not all_crops:
                temp = center_crop(temp, (320,320))
                temp = np.expand_dims(temp,3)
                kspace_train.append(temp)
            else:
                temp0 = center_crop(temp, (320,320), 0)
                temp1 = center_crop(temp, (320,320), -1)
                temp2 = center_crop(temp, (320,320), 1)
                temp0 = np.expand_dims(temp0,3)
                temp1 = np.expand_dims(temp1,3)
                temp2 = np.expand_dims(temp2,3)
                kspace_train.append(temp0)
                kspace_train.append(temp1)
                kspace_train.append(temp2)

    print("concatenating")
    kspace_train = np.concatenate((kspace_train), axis=0)
    print(np.array(kspace_train).shape)
    print(np.array(kspace_train)[0].shape)

    np.savez_compressed('data/kspace_test_center_crops_small', kspace_train=np.array(kspace_train))

## Adding dimension for coil
# kspace_train = np.expand_dims(kspace_train,3)
print(kspace_train.shape)
# kspace_train = center_crop(kspace_train, (320,320),0)
# kspace_train = np.expand_dims(kspace_train,3)

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

broken_mask = True

shape = (args.nrow_GLOB, args.ncol_GLOB)

if not broken_mask:
    kspace_temp = kspace_train.squeeze()
    kspace_temp = transforms.to_tensor(kspace_temp)
else:
    kspace_temp = transforms.to_tensor(kspace_train)
mask = subsample.create_mask_for_mask_type("random", [0.08], [4])
masked_kspace, mask = transforms.apply_mask(kspace_temp, mask)
print(kspace_temp.shape)

original_mask = np.ones(shape)
mask = np.array(mask.squeeze(0).squeeze(0).squeeze(1)).astype(int)
for i in range(mask.shape[0]):
    s = mask[i]
    original_mask[:,i] = s

plt.imsave("mask.png",original_mask, cmap="gray")



tf.logging.info(f'\n size of kspace: {kspace_train.shape}, maps: {sens_maps.shape}, mask: {original_mask.shape}')


trn_mask, loss_mask = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                      np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

nw_input = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
ref_kspace = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
test_refAll = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)


tf.logging.info('\n create training and loss masks and generate network inputs... ')
ssdu_masker = ssdu_masks.ssdu_masks()
for ii in range(nSlices):
    test_refAll[ii] = utils.sense1(kspace_train[ii, ...], sens_maps[ii, ...])

    if np.mod(ii, 50) == 0:
        tf.logging.info(f'\n Iteration: {ii}')

    if args.mask_type == 'Gaussian':
        # tf.logging.info(original_mask)
        trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.Gaussian_selection(kspace_train[ii], original_mask, num_iter=ii)

    elif args.mask_type == 'Uniform':
        trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.uniform_selection(kspace_train[ii], original_mask, num_iter=ii)
        # import ipdb
        # ipdb.set_trace()

    else:
        raise ValueError('Invalid mask selection')

    # print(kspace_train.shape, np.tile(trn_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB)).shape)

    sub_kspace = kspace_train[ii] * np.tile(trn_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    ref_kspace[ii, ...] = kspace_train[ii] * np.tile(loss_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    nw_input[ii, ...] = utils.sense1(sub_kspace, sens_maps[ii, ...])

test_mask = np.complex64(np.tile(original_mask[np.newaxis, :, :], (nSlices, 1, 1)))
print(test_mask.shape, trn_mask.shape, loss_mask.shape)
test_mask[:, :, 0:20] = np.ones((nSlices, args.nrow_GLOB, 20))
test_mask[:, :, 300:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 20))

# %%  zeropadded outer edges of k-space with no signal- check github readme file for explanation for further explanations

trn_mask[:, :, 0:20] = np.ones((nSlices, args.nrow_GLOB, 20))
trn_mask[:, :, 300:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 20))

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
saved_model_dir = "/srv/share4/ksarangmath3/mri/SSDU/saved_models/SSDU_fastmri_knee_small_100Epochs_Rate4_10Unrolls_UniformSelection"
loadChkPoint = tf.train.latest_checkpoint(saved_model_dir)

with tf.Session(config=config) as sess:
    
    # sess.run(tf.global_variables_initializer())

    new_saver = tf.train.import_meta_graph(saved_model_dir + '/model-67.meta')
    new_saver.restore(sess, loadChkPoint)

    tf.logging.info(f'SSDU Parameters: Epochs: {args.epochs} Batch Size: {args.batchSize}, Number of trainable parameters: {sess.run(all_trainable_vars)}, Iterations: {total_batch}')
    if broken_mask:
        feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: trn_mask, loss_maskP: loss_mask, sens_mapsP: sens_maps}
    else:
        feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: test_mask, loss_maskP: test_mask, sens_mapsP: sens_maps}

    tf.logging.info('Training...')
    for ep in range(1, 2):
        sess.run(iterator.initializer, feed_dict=feedDict)
        avg_cost = 0
        tic = time.time()
        SSIM_list = []
        try:
            for jj in range(total_batch):
                tf.logging.info("---------------------------------")
                tmp, out = sess.run([loss, nw_output_img])
                ref_image_test = np.copy(test_refAll[jj, :, :])[np.newaxis]
                ref_image_test = utils.complex2real(ref_image_test)
                ref_image_test = utils.real2complex(ref_image_test.squeeze())
                out = utils.real2complex(out.squeeze())
                ref_image_test = np.abs(ref_image_test)
                out = np.abs(out)
                tf.logging.info(f"Avg Cost : {tmp}")
                print(out.shape, ref_image_test.shape)
                
                SSIM_list.append(utils.getSSIM(ref_image_test, out))
                print(SSIM_list[-1])
                avg_cost += tmp / total_batch
                if jj == 15:
                    plt.imsave(saved_model_dir.split('/')[-1]+"_reconstruction.png",out, cmap="gray")
                    plt.imsave(saved_model_dir.split('/')[-1]+"_groundtruth.png",ref_image_test, cmap="gray")
            toc = time.time() - tic
            totalLoss.append(avg_cost)
            tf.logging.info(f'Epoch: {ep} elapsed_time =""{toc}", "cost =", "{avg_cost}"')
            print(np.mean(SSIM_list), 'FINAL SSIM')
            break
        except tf.errors.OutOfRangeError:
            pass

        # if (np.mod(ep, 1) == 0):
        #     saver.save(sess, sess_trn_filename, global_step=ep)
        #     sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})

end_time = time.time()
sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})
tf.logging.info('Training completed in  ', ((end_time - start_time) / 60), ' minutes')
