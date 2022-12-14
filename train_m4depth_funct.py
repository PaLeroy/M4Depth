from datetime import datetime


from dataloaders.midair import DataLoaderMidAir as MidAir
from dataloaders import DataloaderParameters

import os

import tensorflow as tf
from tensorflow import keras as ks

from m4depth_network_funct import M4Depth
from metrics import AbsRelError, SqRelError, RootMeanSquaredError, \
    RootMeanSquaredLogError, ThresholdRelError

from callbacks import CustomCheckpointCallback

print(tf.__version__)


if __name__ == '__main__':

    param = DataloaderParameters({
    '_comment': '/Users/pascalleroy/Documents/m4depth/M4Depth/relative paths should be written relative to this file',
    'midair': '/home/pascal/m4depth/MidAir',
    'kitti-raw': '/Users/pascalleroy/Documents/m4depth/M4Depth/datasets/Kitti',
    'tartanair': '/Users/pascalleroy/Documents/m4depth/M4Depth/datasets/TartanAir'},
    'data/midair/train_data',
        8,  # db_seq_len
        4,  # seq_len
        True)

    batch_size = 3
    n_lvl = 6
    chosen_dataloader = MidAir()
    chosen_dataloader.get_dataset("train", param, batch_size=batch_size)
    data = chosen_dataloader.dataset

    model = M4Depth(n_levels=n_lvl, old_version = False)
    now = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    tensorboard_cbk = ks.callbacks.TensorBoard(
        log_dir="log_dir/" + now, histogram_freq=chosen_dataloader.length/2, write_graph=False,
        write_images=False, update_freq=chosen_dataloader.length/2)

    weights_dir = os.path.join("pretrained_weights/midair/", "train", now)
    model_checkpoint_cbk = CustomCheckpointCallback(weights_dir, resume_training=True)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, metrics=[RootMeanSquaredLogError()])
    model.save_h5(weights_dir+"/network_save_h5.h5")
    model.save(weights_dir+"/network_save_tf")
    nbre_epochs = (220000 // chosen_dataloader.length)*3

    # model.fit(data, epochs=nbre_epochs, callbacks=[tensorboard_cbk, model_checkpoint_cbk])


    # for idx_data, data in enumerate(data):
    #     print("batch", idx_data)
    #     # iterate over dataset
    #     seq_len = data["depth"].get_shape().as_list()[1]
    #     traj_samples = [{} for i in range(seq_len)]
    #     attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]
    #     for key in attribute_list:
    #         value_list = tf.unstack(data[key], axis=1)
    #         for i, item in enumerate(value_list):
    #             shape = item.get_shape()
    #             traj_samples[i][key] = item
    #     gts = []
    #     for sample in traj_samples:
    #         gts.append({"depth": sample["depth"],
    #                     "disp": depth2disp(sample["depth"],
    #                                        sample["rot"],
    #                                        sample["trans"],
    #                                        data["camera"]["c"],
    #                                        data["camera"]["f"])})
    #     preds = model((traj_samples, data["camera"]))
    #
    #     loss = model.m4depth_loss(gts, preds)
    #     print("loss: ", loss)
    #     tf.print("loss: ", loss)