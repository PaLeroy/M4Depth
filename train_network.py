from dataloaders.midair import DataLoaderMidAir as MidAir
from dataloaders import DataloaderParameters

param = DataloaderParameters({'_comment': '/Users/pascalleroy/Documents/m4depth/M4Depth/relative paths should be written relative to this file',
                                 'midair': '/Users/pascalleroy/Documents/m4depth/M4Depth/datasets/MidAir',
                                 'kitti-raw': '/Users/pascalleroy/Documents/m4depth/M4Depth/datasets/Kitti',
                                 'tartanair': '/Users/pascalleroy/Documents/m4depth/M4Depth/datasets/TartanAir'},
                             'data/midair/small_train_data',
                             8, #db_seq_len
                             4, # seq_len
                             True)
print(param)
chosen_dataloader = MidAir()
chosen_dataloader.get_dataset("train", param, batch_size=4)
data = chosen_dataloader.dataset

from importlib import reload
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras as ks
from keras.utils import vis_utils
from metrics import AbsRelError, SqRelError, RootMeanSquaredError, \
    RootMeanSquaredLogError, ThresholdRelError

from callbacks import CustomCheckpointCallback

print(tf.__version__)
from m4depth_network import M4Depth, M4depthAblationParameters

tf.random.set_seed(42)

ablation_settings = M4depthAblationParameters()

model = M4Depth(depth_type=chosen_dataloader.depth_type,
                nbre_levels=6,
                ablation_settings=ablation_settings,
                is_training=True)

# Initialize callbacks
tensorboard_cbk = ks.callbacks.TensorBoard(
    log_dir="log_dir", histogram_freq=1200, write_graph=True,
    write_images=False, update_freq=1200,
    profile_batch=0, embeddings_freq=0, embeddings_metadata=None)
weights_dir = os.path.join("pretrained_weights/midair/", "train")
model_checkpoint_cbk = CustomCheckpointCallback(weights_dir, resume_training=True)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=opt, metrics=[RootMeanSquaredLogError()])
nbre_epochs = 2

model.fit(data, epochs= nbre_epochs + 1,
          initial_epoch=model_checkpoint_cbk.resume_epoch,
          callbacks=[tensorboard_cbk, model_checkpoint_cbk])