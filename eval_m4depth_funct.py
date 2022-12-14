from metrics import AbsRelError, SqRelError, RootMeanSquaredError, RootMeanSquaredLogError, ThresholdRelError
from dataloaders.midair import DataLoaderMidAir as MidAir
from dataloaders import DataloaderParameters
import os
import tensorflow as tf
from tensorflow import keras as ks
from m4depth_network_funct import M4Depth
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

    dir_name = "12_02_2022_16_55_20"
    weights_dir = os.path.join("pretrained_weights/midair/", dir_name)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="log_dir/test_" + dir_name, profile_batch='10, 25')
    # tensorboard_cbk = ks.callbacks.TensorBoard(
    #     log_dir="log_dir/" + now, histogram_freq=chosen_dataloader.length/2, write_graph=False,
    #     write_images=False, update_freq=chosen_dataloader.length/2)

    model = M4Depth(n_levels=n_lvl, old_version = False)
    model_checkpoint_cbk = CustomCheckpointCallback(weights_dir, resume_training=True)
    model.compile(metrics=[AbsRelError(),
                           SqRelError(),
                           RootMeanSquaredError(),
                           RootMeanSquaredLogError(),
                           ThresholdRelError(1), ThresholdRelError(2), ThresholdRelError(3)])

    metrics = model.evaluate(data, callbacks=[model_checkpoint_cbk])
    print(metrics)
    # # Keep track of the computed performance
    # if cmd.mode == 'validation':
    #     manager = BestCheckpointManager(os.path.join(ckpt_dir, "train"), os.path.join(ckpt_dir, "best"),
    #                                     keep_top_n=cmd.keep_top_n)
    #     perfs = {"abs_rel": [metrics[0]], "sq_rel": [metrics[1]], "rmse": [metrics[2]], "rmsel": [metrics[3]],
    #              "a1": [metrics[4]], "a2": [metrics[5]], "a3": [metrics[6]]}
    #     manager.update_backup(perfs)
    #     string = ''
    #     for perf in metrics:
    #         string += format(perf, '.4f') + "\t\t"
    #     with open(os.path.join(*[ckpt_dir, "validation-perfs.txt"]), 'a') as file:
    #         file.write(string + '\n')
    # else:
    #     np.savetxt(os.path.join(*[ckpt_dir, "perfs-" + cmd.dataset + ".txt"]), metrics, fmt='%.18e', delimiter='\t',
    #                newline='\n')