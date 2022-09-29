from dataloaders.midair import DataLoaderMidAir as MidAir
from dataloaders import DataloaderParameters

param = DataloaderParameters({'_comment': '/Users/pascalleroy/Documents/m4depth/M4Depth/relative paths should be written relative to this file', 'midair': '/Users/pascalleroy/Documents/m4depth/M4Depth/datasets/MidAir', 'kitti-raw': '/Users/pascalleroy/Documents/m4depth/M4Depth/datasets/Kitti', 'tartanair': '/Users/pascalleroy/Documents/m4depth/M4Depth/datasets/TartanAir'},
    'data/midair/small_test_data',
    None,
    4,
    True)

chosen_dataloader = MidAir()
chosen_dataloader.get_dataset("eval", param, batch_size=1)
data = chosen_dataloader.dataset

from importlib import reload
import os
import tensorflow as tf
from tensorflow import keras as ks
from keras.utils import vis_utils
from metrics import AbsRelError, SqRelError, RootMeanSquaredError, \
    RootMeanSquaredLogError, ThresholdRelError

from callbacks import CustomCheckpointCallback

print(tf.__version__)
from m4depth_simplified_network import M4Depth


print(tf.config.list_physical_devices('GPU'))
model = M4Depth(nbre_levels=6)

weights_dir = os.path.join("pretrained_weights/midair/", "best")
model_checkpoint_cbk = CustomCheckpointCallback(weights_dir,
                                                resume_training=True)
model.compile(metrics=[AbsRelError(),
                       SqRelError(),
                       RootMeanSquaredError(),
                       RootMeanSquaredLogError(),
                       ThresholdRelError(1), ThresholdRelError(2),
                       ThresholdRelError(3)])

metrics = model.evaluate(data, callbacks=[model_checkpoint_cbk])
print(metrics)
print(model.summary())


# weights_dir = os.path.join("pretrained_weights/midair/","best")
# model_checkpoint_cbk = CustomCheckpointCallback(weights_dir, resume_training=True)
# model.compile()
#
# #metrics = model.evaluate(data, callbacks=[model_checkpoint_cbk])
#
# first_sample = data.take(3)
# a = model.predict(first_sample, callbacks=[model_checkpoint_cbk], verbose=2)

# is_first_run = True
#
# # Do what you want with the outputs
# for i, sample in enumerate(data):
#     if not is_first_run and sample["new_traj"]:
#         print("End of trajectory")
#     print(i)
#     is_first_run = False
#     est = model([[sample], sample["camera"]]) # Run network to get estimates
#     d_est = est["depth"][0, :, :, :]        # Estimate : [h,w,1] matrix with depth in meter
#     d_gt = sample['depth'][0, :, :, :]      # Ground truth : [h,w,1] matrix with depth in meter
#     i_rgb = sample['RGB_im'][0, :, :, :]    # RGB image : [h,w,3] matrix with rgb channels ranging between 0 and 1
