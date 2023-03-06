import tensorflow as tf
from tensorflow import keras as ks, float32

from metrics import RootMeanSquaredLogError
from utils.depth_operations_functional import get_disparity_sweeping_cv, \
    prev_d2disp, disp2depth, cost_volume, \
    depth2disp, lambda_disp2depth, lambda_prev_d2disp, \
    lambda_get_disparity_sweeping_cv, lambda_cost_volume
from utils.depth_operations import \
    reproject  # We do not need the functionnal reproject
import numpy as np

class RescaleLayer(ks.layers.Layer):
    def __init__(self, regularizer_weight=0.0004, *args, **kwargs):
        super(RescaleLayer, self).__init__(*args, **kwargs)
        self.regularizer_weight = regularizer_weight

    def build(self, input_shape):
        channels = input_shape[-1]

        self.scale = self.add_weight(name="scale", shape=[1, 1, 1, channels],
                                     dtype='float32',
                                     initializer=tf.ones_initializer(),
                                     trainable=True)
        self.bias = self.add_weight(name="bias", shape=[1, 1, 1, channels],
                                    dtype='float32',
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
        # Add regularization loss on the scale factor
        regularizer = tf.keras.regularizers.L2(self.regularizer_weight)
        self.add_loss(regularizer(self.scale))

    def call(self, f_map):
        return self.scale * f_map + self.bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "regularizer_weight": self.regularizer_weight,
        })
        return config

def domain_normalization_as_a_function(f_map, regularizer_weight, name):
    # Normalizes a feature map according to the procedure presented by
    # Zhang et.al. in "Domain-invariant stereo matching networks".
    # Also see DomainNormalization in former implem of M4Depth.
    # ks.layers.Layer.add_loss(regularizer(scale))
    channels = f_map.shape[-1]

    mean = tf.math.reduce_mean(f_map, axis=[1, 2], keepdims=True, name=None)
    var = tf.math.reduce_variance(f_map, axis=[1, 2], keepdims=True, name=None)
    normed = tf.math.l2_normalize((f_map - mean) / (var + 1e-12), axis=-1)
    normed = RescaleLayer(regularizer_weight, name=name)(normed)

    # This does not work
    # scale = tf.Variable(trainable=True,
    #                     initial_value=tf.ones((1, 1, 1, channels)),
    #                     dtype=float32)
    # bias = tf.Variable(trainable=True,
    #                    initial_value=tf.zeros((1, 1, 1, channels)),
    #                    dtype=float32)
    # normed = scale * normed + bias
    #
    return normed


@tf.function
def log_tensor_companion(x):
    count = tf.math.count_nonzero(tf.math.is_nan(x[0]))
    if count != 0:
        tf.print("nan found", x[1])
    return x


def log_tensor(x, name=""):
    return ks.layers.Lambda(log_tensor_companion)((x, name))[0]


def disp_refiner_as_a_function(regularizer_weight, name, feature_map):
    init = ks.initializers.HeNormal()
    reg = ks.regularizers.L1(l1=regularizer_weight)
    conv_channels = [128, 128, 96]
    prep_conv_layers = [ks.layers.Conv2D(
        nbre_filters, 3, strides=(1, 1), padding='same',
        kernel_initializer=init, kernel_regularizer=reg,
        name=name + "_prep_conv_layers_" + str(i))
        for i, nbre_filters in enumerate(conv_channels)
    ]
    conv_channels = [64, 32, 16, 5]
    est_d_conv_layers = [ks.layers.Conv2D(
        nbre_filters, 3, strides=(1, 1), padding='same',
        kernel_initializer=init, kernel_regularizer=reg,
        name=name + "_est_d_conv_layers" + str(i))
        for i, nbre_filters in enumerate(conv_channels)
    ]
    prev_out = tf.identity(feature_map)
    # prev_out = log_tensor(prev_out, "prev_out_before")
    for i, conv in enumerate(prep_conv_layers):
        # prev_out = log_tensor(prev_out, name + "just before before prev_out_before conv" + str(i))
        prev_out = conv(prev_out)
        # prev_out = log_tensor(prev_out, name + "prev_out_before conv" + str(i))
        prev_out = ks.layers.LeakyReLU(0.1,
                                       name=name + "_prep_conv_layers_ReLu_" + str(
                                           i))(prev_out)
        # prev_out = log_tensor(prev_out, "prev_out_before leaky" + str(i))

    prev_outs = [prev_out, prev_out]

    for i, convs in enumerate(zip(est_d_conv_layers)):

        for j, (prev, conv) in enumerate(zip(prev_outs, convs)):
            prev_outs[j] = conv(prev)
            if i < len(
                    est_d_conv_layers) - 1:  # Don't activate last convolution output
                prev_outs[j] = ks.layers.LeakyReLU(0.1,
                                                   name=name + "_est_d_conv_layers_ReLu_" + str(
                                                       i))(prev_outs[j])
    return prev_outs  # tf.concat(prev_outs, axis=-1)


def M4Depth_functionnal_model(nbre_levels=6, regularizer_weight=0.0004):
    """
    Functionnal version of M4Depth
    """

    # We start with all the inputs of the network graph from the dataset

    img_input = ks.Input(shape=(384, 384, 3,), dtype=float32,
                         name="image")  # data image
    inputs_list = [img_input]

    camera_f_input = ks.Input(shape=(2,), dtype=float32,
                              name="camera_f_input")  # data camera property
    inputs_list.append(camera_f_input)

    camera_c_input = ks.Input(shape=(2,), dtype=float32,
                              name="camera_c_input")  # data camera property
    inputs_list.append(camera_c_input)

    rot_input = ks.Input(shape=(4,), dtype=float32,
                         name="rot_input")  # data camera displacement
    inputs_list.append(rot_input)

    trans_input = ks.Input(shape=(3,), dtype=float32,
                           name="trans_input")  # data camera displacement
    inputs_list.append(trans_input)

    # new_traj_input = ks.Input(shape=(1,), dtype=bool,
    #                           name="new_traj_input")  # data is it the first elem of a sequence
    # inputs_list.append(new_traj_input)
    # depth_input = ks.Input(shape=(384, 384, 3,), dtype=float32,
    #                        name="depth_input_ground_truth")  # groundtruth
    # inputs_list.append(depth_input)

    encoder_output_list_per_level = []

    init = ks.initializers.HeNormal()
    reg = ks.regularizers.L1(l1=regularizer_weight)

    all_filter_sizes = [16, 32, 64, 96, 128, 192]
    filter_sizes = [all_filter_sizes[i] for i in range(nbre_levels)]
    x = img_input

    for idx, n_filter in enumerate(filter_sizes):
        layer_string = "L_" + str(idx + 1)
        conv_layers_s1_output = ks.layers.Conv2D(
            n_filter, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg,
            name=layer_string + "_Encoder_s1")(x)
        x = ks.layers.LeakyReLU(0.1,
                                name=layer_string + "_Encoder_s1_LeakyReLU")(
            conv_layers_s1_output)

        if idx == 0:
            # x = DomainNormalization(regularizer_weight=regularizer_weight,
            #                             name=layer_string + "_Encoder_DN")(x)
            x = domain_normalization_as_a_function(x,
                                                   regularizer_weight,
                                                   name=layer_string+"_Encoder_DN")
            # print(x.shape)
        conv_layers_s2_output = ks.layers.Conv2D(
            n_filter, 3, strides=(2, 2), padding='same',
            kernel_initializer=init, kernel_regularizer=reg,
            name=layer_string + "_Encoder_s2")(x)
        x = ks.layers.LeakyReLU(0.1, name=layer_string + "_f_enc_L_t")(
            conv_layers_s2_output)
        # x = log_tensor(x, layer_string + "_f_enc_L_t")
        encoder_output_list_per_level.append(x)

    all_output_list = []

    # The decoder requires some inputs at its lower level
    b, h, w, c = encoder_output_list_per_level[-1].shape

    disp_L1_t_input = ks.Input(shape=(h, w, 1,), dtype=float32,
                               name=layer_string + "_disp_L-1_t")
    inputs_list.append(disp_L1_t_input)

    other_L1_t_input = ks.Input(shape=(h, w, 4,), dtype=float32,
                                name=layer_string + "_other_L-1_t")
    inputs_list.append(other_L1_t_input)

    # We start with the bottom layer of the encoder,
    # that's why we flip the list.
    # Reminder about layer: 1
    d_est_all_levels = []
    custom_out_list = []
    for l_index, f_enc_L_t in enumerate(encoder_output_list_per_level[::-1]):

        lvl_depth = nbre_levels - l_index  # lvl_depth = L in the paper
        lvl_mul = lvl_depth - 3
        layer_string = "L_" + str(lvl_depth)
        prev_layer_string = "L_" + str(lvl_depth + 1)
        b, h, w, c = f_enc_L_t.shape

        # l_index = 0 is the smallest image of the decoder
        # divide_expo_function = lambda x: x / 2. ** float(lvl_depth)
        local_camera_f = camera_f_input / 2. ** float(lvl_depth)
        local_camera_c = camera_c_input / 2. ** float(lvl_depth)

        # rot and trans unchanged because everything else is rescaled
        local_rot = rot_input
        local_trans = trans_input

        # input from the encoder at the same level at the same timestep
        # = curr_f_maps in DepthEstimatorLevel
        f_enc_L_t = f_enc_L_t
        # input from the encoder at the same level at the previous timestep
        # t1 rps t-1
        # = prev_f_maps in DepthEstimatorLevel
        f_enc_L_t1 = ks.Input(shape=(h, w, c,), dtype=float32,
                              name=layer_string + "_f_enc_L_t-1")
        inputs_list.append(f_enc_L_t1)

        # input from the disparity refiner at the same level
        # at the previous timestep
        # = prev_t_depth in DepthEstimatorLevel
        depth_L_t1 = ks.Input(shape=(h, w, 1,), dtype=float32,
                              name=layer_string + "_d_est_L_t-1")
        inputs_list.append(depth_L_t1)

        # input from the disparity refiner at the previous level
        # at the same timestep
        # L1 rps the previous (lower) level L-1
        # = prev_l_est in DepthEstimatorLevel
        if l_index == 0:
            disp_L1_t = disp_L1_t_input
            other_L1_t = other_L1_t_input
        else:
            # resize_bilinear_function =
            # lambda x2: tf.compat.v1.image.resize_bilinear(x2, [h, w])
            # resize_bilinear_function =
            # lambda x2: tf.image.resize(x2, [h, w])
            # disp_L1_t = ks.layers.Lambda(resize_bilinear_function,
            # name = layer_string+ "_disp_L-1_t_resized")
            # (d_est_all_levels[-1]["disp"])
            # depth_L1_t = ks.layers.Lambda(resize_bilinear_function,
            # name = layer_string+ "_depth_L-1_t_resized")
            # (d_est_all_levels[-1]["depth"])
            # other_L1_t = ks.layers.Lambda(resize_bilinear_function,
            # name = layer_string+ "_other_L-1_t_resized")
            # (d_est_all_levels[-1]["other"])

            disp_L1_t = d_est_all_levels[-1][prev_layer_string + "_disp"]
            disp_L1_t = tf.image.resize(disp_L1_t, [h, w],
                                        name=layer_string
                                             + "_disp_L-1_t_resized")

            other_L1_t = d_est_all_levels[-1][prev_layer_string + "_other"]
            other_L1_t = tf.image.resize(other_L1_t, [h, w],
                                         name=layer_string
                                              + "_other_L-1_t_resized")

        nbre_cuts = 2 ** (lvl_depth // 2)

        f_enc_L_t = ks.layers.Reshape((h, w, nbre_cuts, -1,),
                                      name=layer_string + "_cuts")(
            f_enc_L_t)

        #  ks.utils.normalize = tf.linalg.normalize is L2 norm = euclidian norm
        # TODO: check with ks.layer.normalization
        # norm_function = lambda x: tf.math.l2_normalize(x, axis=-1)
        # f_enc_L_t = ks.layers.Lambda(norm_function,
        # name=layer_string+ "_normalise_cuts")(f_enc_L_t)
        f_enc_L_t = tf.math.l2_normalize(f_enc_L_t, axis=-1,
                                         name=layer_string + "_normalise_cuts")
        # f_enc_L_t = log_tensor(f_enc_L_t, layer_string + 'f_enc_L_t l2_normalize')
        f_enc_L_t = ks.layers.Reshape((h, w, -1,),
                                      name=layer_string + "_concat_cuts")(
            f_enc_L_t)
        # f_enc_L_t = log_tensor(f_enc_L_t, layer_string + 'f_enc_L_t')

        # depth_L_t1 = log_tensor(depth_L_t1, layer_string + ' depth_L_t1')
        # if old_version:
        #     prev_d2disp_function = \
        #         lambda inp: lambda_prev_d2disp(inp[0], inp[1], inp[2], inp[3], inp[4])
        #     prev_d2disp_layer = ks.layers.Lambda(prev_d2disp_function,
        #                                          name=layer_string
        #                                               + "_prev_d2disp")
        #     disp_L_t1 = prev_d2disp_layer((depth_L_t1, local_rot, local_trans,
        #                                    local_camera_c, local_camera_f))
        # else:
        disp_L_t1 = prev_d2disp(depth_L_t1, local_rot, local_trans,
                                           local_camera_c, local_camera_f)

        # disp_L_t1 = log_tensor(disp_L_t1, layer_string + ' disp_L_t1')
        # disp_L_t1 = prev_d2disp(depth_L_t1, local_rot,
        #                         local_trans, local_camera_c, local_camera_f)

        # get_disparity_sweeping_cv_function = lambda \
        #         inp: get_disparity_sweeping_cv(inp[0], inp[1], inp[2], inp[3],
        #                                        inp[4], inp[5], inp[6], inp[7])
        # TODO: nbre_cut must be a tensor part of the input?
        # if old_version:
        #     get_disparity_sweeping_cv_layer = tf.keras.layers.Lambda(
        #         lambda_get_disparity_sweeping_cv,
        #         arguments={"search_range": 4, "nbre_cuts": nbre_cuts},
        #         name=layer_string + "_get_disparity_sweeping_cv")
        #     cv, disp_prev_t_reproj = \
        #         get_disparity_sweeping_cv_layer((f_enc_L_t, f_enc_L_t1, disp_L_t1,
        #                                          disp_L1_t, local_rot, local_trans,
        #                                          local_camera_c, local_camera_f,))
        # else:
        cv, disp_prev_t_reproj = get_disparity_sweeping_cv(
                    (f_enc_L_t, f_enc_L_t1, disp_L_t1,
                     disp_L1_t, local_rot, local_trans,
                     local_camera_c, local_camera_f),
                search_range= 4,
                nbre_cuts=nbre_cuts)
        # cv = log_tensor(cv, layer_string + 'cv')
        # cv, disp_prev_t_reproj = \
        #     get_disparity_sweeping_cv(f_enc_L_t, f_enc_L_t1, disp_L_t1,
        #                               disp_L1_t, local_rot, local_trans,
        #                               local_camera_c, local_camera_f,
        #                               4, nbre_cuts)
        # if old_version:
        #     autocorr = ks.layers.Lambda(lambda_cost_volume,
        #                                 arguments={"search_range": 3,
        #                                            "nbre_cuts": nbre_cuts},
        #                                 name=layer_string + "_autocorr_function") \
        #         (f_enc_L_t)
        # else:
        autocorr = cost_volume(f_enc_L_t, search_range=3, nbre_cuts=nbre_cuts)
        # autocorr = log_tensor(autocorr, layer_string + 'autocorr')

        # autocorr = cost_volume(f_enc_L_t, f_enc_L_t, 3,
        # nbre_cuts=nbre_cuts, name=layer_string+"_autocorr_function")

        # disp_prev_t_reproj_log_function
        # = lambda x : tf.math.log(x[:,:,:,4:5]*2**lvl_mul)
        # disp_prev_t_reproj = ks.layers.Lambda
        # (disp_prev_t_reproj_log_function,
        # name=layer_string+"disp_prev_t_reproj_log_function")
        # (disp_prev_t_reproj)

        disp_prev_t_reproj = tf.math.log(
            disp_prev_t_reproj[:, :, :, 4:5] * 2 ** lvl_mul,
            name=layer_string + "disp_prev_t_reproj_log_function")
        # disp_prev_t_reproj = log_tensor(disp_prev_t_reproj, layer_string + 'disp_prev_t_reproj')
        # disp_L1_t_log_function = lambda x: tf.math.log(x*2**lvl_mul)
        # disp_L1_t = ks.layers.Lambda(disp_L1_t_log_function,
        # name=layer_string+"_disp_L-1_t_log_function")(disp_L1_t)
        disp_L1_t = tf.math.log(disp_L1_t * 2 ** lvl_mul,
                                name=layer_string + "_disp_L-1_t_log_function")
        # disp_L1_t = log_tensor(disp_L1_t, layer_string + 'disp_L1_t')

        f_input = \
            ks.layers.Concatenate(
                axis=-1,
                name=layer_string + "_Concatenate_cv_disp_L-1_t")(
                [cv, disp_L1_t])
        f_input = \
            ks.layers.Concatenate(
                axis=-1,
                name=layer_string + "_Concatenate_other_L-1_t")(
                [f_input, other_L1_t])
        f_input = \
            ks.layers.Concatenate(
                axis=-1,
                name=layer_string + "_Concatenate_autocorr")(
                [f_input, autocorr])
        f_input = \
            ks.layers.Concatenate(
                axis=-1,
                name=layer_string + "_Concatenate_disp_prev_t_reproj")(
                [f_input, disp_prev_t_reproj])

        # f_input = log_tensor(f_input,layer_string +  'f_input')
        prev_out = disp_refiner_as_a_function(
            regularizer_weight=regularizer_weight,
            name=layer_string + "_disp_refiner", feature_map=f_input)

        # slicing_disp_function = lambda x: x[0][:, :, :, :1]
        # disp = ks.layers.Lambda(slicing_disp_function,
        # name=layer_string + "_slicing_disp_function")(prev_out)
        disp = prev_out[0][:, :, :, :1]
        # slicing_other_function = lambda x: x[0][:, :, :, 1:]
        # other = ks.layers.Lambda(slicing_other_function,
        # name=layer_string + "_slicing_other_function")(prev_out)
        other = prev_out[0][:, :, :, 1:]
        # exp_clip_function = lambda x:
        # tf.exp(tf.clip_by_value(x, -7., 7.))/2**lvl_mul
        # disp_curr_l = ks.layers.Lambda(exp_clip_function,
        # name=layer_string + "_disp_curr_l_exp_clip_function")(disp)
        disp_curr_l = \
            tf.exp(tf.clip_by_value(disp, -7., 7.)) / 2 ** lvl_mul

        # if old_version:
        #     disp2depth_function = \
        #         lambda x: lambda_disp2depth(x[0], x[1], x[2], x[3], x[4])
        #     disp2depth_function_layer = ks.layers.Lambda(disp2depth_function,
        #                                                  name=layer_string + "_depth_curr_l_disp2depth_function")
        #     depth_curr_l = disp2depth_function_layer(
        #         (disp_curr_l, local_rot, local_trans,
        #          local_camera_c, local_camera_f))
        # else:
        depth_curr_l = disp2depth(disp_curr_l, local_rot, local_trans,
                                      local_camera_c, local_camera_f)

        # These layers have no effect on value directly
        # but allow to isolate and rename variables
        other_output = ks.layers.Layer(
            name=layer_string + "_other_identity") \
            (tf.identity(other))
        depth_output = ks.layers.Layer(
            name=layer_string + "_depth_identity") \
            (tf.identity(depth_curr_l))
        disp_output = ks.layers.Layer(
            name=layer_string + "_disp_identity") \
            (tf.identity(disp_curr_l))

        # other_output = log_tensor(other_output, layer_string + 'other_output')
        # depth_output = log_tensor(depth_output, layer_string + 'depth_output')
        # disp_output = log_tensor(disp_output, layer_string + 'disp_output')

        curr_l_est = {
            layer_string + "_other": other_output,
            layer_string + "_depth": depth_output,
            layer_string + "_disp": disp_output
        }

        d_est_all_levels.append(curr_l_est)

    all_output_list = []
    for cpt, i in enumerate(d_est_all_levels):
        for k, v in i.items():
            all_output_list.append(v)

    return ks.Model(inputs=inputs_list,
                    outputs=encoder_output_list_per_level + all_output_list), all_filter_sizes


class M4Depth(ks.models.Model):
    """Tensorflow model of M4Depth"""

    def __init__(self, n_levels=6, regularizer_weight=0.):
        super(M4Depth, self).__init__()
        regularizer_weight = 0.

        self.full_model, self.all_filter_sizes = M4Depth_functionnal_model(
            n_levels,
            regularizer_weight=regularizer_weight)
        self.n_levels = n_levels
        self.h = 384
        self.w = 384
        self.step_counter = tf.Variable(
            initial_value=tf.zeros_initializer()(shape=[], dtype='int64'),
            trainable=False)
        self.summaries = []

        self.batch_size = 1

    def inputs_init_seq(self, batch_size):
        # Inputs for the beginning of a sequence -> once per seq
        # Starting from the bottom of the U -> (L3, L2, L1)
        dict_inputs = {}
        h = self.h
        w = self.w
        for i in range(self.n_levels):
            h = int(h / 2)
            w = int(w / 2)
            last_dim = self.all_filter_sizes[i]
            dict_inputs["L_" + str(i + 1) + "_f_enc_L_t1_init"] = tf.zeros(
                (batch_size, h, w, last_dim), dtype=float32)
            dict_inputs["L_" + str(i + 1) + "_d_est_L_t1_init"] = tf.ones(
                (batch_size, h, w, 1), dtype=float32)

        return dict_inputs

    def inputs_init_step(self, batch_size):
        # Inputs for the bottom of the U net -> at each call
        # TODO: maybe possible to hardcode it in the network?
        h = int(self.h / (2 ** self.n_levels))
        w = int(self.w / (2 ** self.n_levels))
        disp_L1_t_init = tf.ones((batch_size, h, w, 1), dtype=float32)
        other_L1_t_init = tf.zeros((batch_size, h, w, 4), dtype=float32)

        return {"disp_L1_t_init": disp_L1_t_init,
                "other_L1_t_init": other_L1_t_init}

    @tf.function
    def call(self, data):
        # data = traj_sample, camera, inputs_recurrent
        self.step_counter.assign_add(1)
        # Traj samples items are [batch_size, seq_len, ....]
        traj_samples = data[0]
        camera = data[1]
        inputs_recurrent=data[2].copy()
        depths = []
        camera_c_input = camera["c"]
        camera_f_input = camera["f"]

        # inputs_recurrent = [f, d, f, d, f, d] starting at the lowest level
        batch_size = traj_samples[0]["rot"].shape[0]

        for idx_sample, sample in enumerate(traj_samples):
            # for each elem of the sequence
            image = sample["RGB_im"]
            rot_input = sample["rot"]
            trans_input = sample["trans"]

            inputs_init_step = self.inputs_init_step(batch_size)
            disp_L1_t = inputs_init_step["disp_L1_t_init"]
            other_L1_t = inputs_init_step["other_L1_t_init"]

            inputs = [image, camera_f_input, camera_c_input, rot_input,
                      trans_input, disp_L1_t, other_L1_t] + inputs_recurrent

            outputs = self.full_model(inputs)
            # outputs= [encoder, decoder]
            # encoder = one per level, starting from the highest
            # decoder = three per level, starting from the lowest

            # first update the recurrent inputs with the encoder
            for i in range(self.n_levels):
                # outputs[i] is L_(i+1)_f_enc_L_t1
                # -> inputs_recurrent[(n-i-1)*2]
                inputs_recurrent[(self.n_levels - i - 1) * 2] = outputs[i]
            list_depth = []
            for i in range(self.n_levels):
                j = i * 3 + 1
                # outputs[n+i*3] is other, outputs_[n+i*3+2] is disp
                # outputs[n+i*3+1] is L_(i+1)_d_est_L_t1
                inputs_recurrent[i * 2 + 1] = outputs[self.n_levels + j]
                list_depth.append({"depth": outputs[self.n_levels + j]})

            # Depth list is lower level first.
            # We need lower level last to act like former implementation.
            # We want (list_depth[0] = pred for the image size)
            # TODO: possible when we fill list_depth but fastest solution to code
            depths.append(list_depth[::-1])

        # for idx, i in enumerate(depths):
        #     for idx2, j in enumerate(i):
        #         for k, v in j.items():
        #             print(idx, idx2, k, v.shape)
        return depths

    @tf.function
    def train_step(self, data):
        with tf.name_scope("train_scope"):
            with tf.GradientTape() as tape:
                # iterate over dataset
                seq_len = data["depth"].get_shape().as_list()[1]
                batch_size = data["depth"].get_shape().as_list()[0]
                traj_samples = [{} for i in range(seq_len)]
                attribute_list = ["depth", "RGB_im", "new_traj", "rot",
                                  "trans"]
                for key in attribute_list:
                    value_list = tf.unstack(data[key], axis=1)
                    for i, item in enumerate(value_list):
                        traj_samples[i][key] = item
                gts = []
                for sample in traj_samples:
                    gts.append({"depth": sample["depth"],
                                "disp": depth2disp(sample["depth"],
                                                   sample["rot"],
                                                   sample["trans"],
                                                   data["camera"]["c"],
                                                   data["camera"]["f"])})


                inputs_init_seq = self.inputs_init_seq(batch_size)
                inputs_recurrent = []
                for i in range(self.n_levels, 0, -1):
                    key = "L_" + str(i) + "_f_enc_L_t1_init"
                    inputs_recurrent.append(inputs_init_seq[key])
                    key = "L_" + str(i) + "_d_est_L_t1_init"
                    inputs_recurrent.append(inputs_init_seq[key])

                preds = self((traj_samples, data["camera"], inputs_recurrent))

                loss = self.m4depth_loss(gts, preds)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # for weights, grads in zip(self.full_model.trainable_weights,
            #                           gradients):
            #     count = tf.math.count_nonzero(tf.math.is_nan(grads))
            #     if count != 0:
            #         tf.print("nan found gradients",
            #                  weights.name.replace(':', '_') + '_grads')
            #
            #     count = tf.math.count_nonzero(tf.math.is_nan(weights))
            #     if count != 0:
            #         tf.print("nan found weights",
            #                  weights.name.replace(':', '_') + '_weights')
            #
            #     tf.summary.histogram(weights.name.replace(':', '_') + '_grads',
            #                          data=grads, step=self.step_counter)
            #     abs_grad = tf.abs(grads)
            #     tf.summary.scalar(
            #         weights.name.replace(':', '_') + '_grads_mean',
            #         data=tf.reduce_mean(abs_grad), step=self.step_counter)
            #     tf.summary.scalar(
            #         weights.name.replace(':', '_') + '_grads_min',
            #         data=tf.reduce_min(abs_grad), step=self.step_counter)
            #     tf.summary.scalar(
            #         weights.name.replace(':', '_') + '_grads_max',
            #         data=tf.reduce_max(abs_grad), step=self.step_counter)
            #
            #     tf.summary.histogram(
            #         weights.name.replace(':', '_') + '_weights', data=weights,
            #         step=self.step_counter)
            #     tf.summary.scalar(
            #         weights.name.replace(':', '_') + '_weights_mean',
            #         data=tf.reduce_mean(weights),
            #         step=self.step_counter)
            #     tf.summary.scalar(
            #         weights.name.replace(':', '_') + '_weights_min',
            #         data=tf.reduce_min(weights),
            #         step=self.step_counter)
            #     tf.summary.scalar(
            #         weights.name.replace(':', '_') + '_weights_max',
            #         data=tf.reduce_max(weights),
            #         step=self.step_counter)

            # Update metrics (includes the metric that tracks the loss)

        with tf.name_scope("summaries"):
            max_d = 200.
            gt_d_clipped = tf.clip_by_value(traj_samples[-1]['depth'], 1.,
                                            max_d)
            tf.summary.image("RGB_im", traj_samples[-1]['RGB_im'],
                             step=self.step_counter)
            im_reproj, _ = reproject(traj_samples[-2]['RGB_im'],
                                     traj_samples[-1]['depth'],
                                     traj_samples[-1]['rot'],
                                     traj_samples[-1]['trans'], data["camera"])
            tf.summary.image("camera_prev_t_reproj", im_reproj,
                             step=self.step_counter)

            tf.summary.image("depth_gt",
                             tf.math.log(gt_d_clipped) / tf.math.log(max_d),
                             step=self.step_counter)
            for i, est in enumerate(preds[-1]):
                d_est_clipped = tf.clip_by_value(est["depth"], 1., max_d)
                self.summaries.append(
                    [tf.summary.image, "depth_lvl_%i" % i,
                     tf.math.log(d_est_clipped) / tf.math.log(max_d)])
                tf.summary.image("depth_lvl_%i" % i,
                                 tf.math.log(d_est_clipped) / tf.math.log(
                                     max_d),
                                 step=self.step_counter)

        with tf.name_scope("metrics"):
            gt = gts[-1]["depth"]
            est = tf.image.resize(preds[-1][0]["depth"], gt.get_shape()[1:3],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            max_d = 80.
            gt = tf.clip_by_value(gt, 0.00, max_d)
            est = tf.clip_by_value(est, 0.001, max_d)
            self.compiled_metrics.update_state(gt, est)
            out_dict = {m.name: m.result() for m in self.metrics}
            out_dict["loss"] = loss

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return out_dict

    @tf.function
    # def test_step(self, data):
    #     # expects one sequence element at a time (batch dim required and is free to set)"
    #     data_format = len(data["depth"].get_shape().as_list())
    #
    #     # If sequence was received as input, compute performance metrics only on its last frame (required for KITTI benchmark))
    #     if data_format == 5:
    #         seq_len = data["depth"].get_shape().as_list()[1]
    #         traj_samples = [{} for i in range(seq_len)]
    #         attribute_list = ["depth", "RGB_im", "new_traj", "rot", "trans"]
    #         for key in attribute_list:
    #             value_list = tf.unstack(data[key], axis=1)
    #             for i, item in enumerate(value_list):
    #                 shape = item.get_shape()
    #                 traj_samples[i][key] = item
    #
    #         gts = []
    #         for sample in traj_samples:
    #             gts.append({"depth": sample["depth"],
    #                         "disp": depth2disp(sample["depth"], sample["rot"],
    #                                            sample["trans"],
    #                                            data["camera"])})
    #         preds = self([traj_samples, data["camera"]], training=False)
    #         gt = data["depth"][:, -1, :, :, :]
    #         est = preds["depth"]
    #         new_traj = False
    #     else:
    #         preds = self([[data], data["camera"]], training=False)
    #         gt = data["depth"]
    #         est = preds["depth"]
    #         new_traj = data["new_traj"]
    #
    #     with tf.name_scope("metrics"):
    #         # Compute performance scores
    #
    #         max_d = 80.
    #         gt = tf.clip_by_value(gt, 0.0, max_d)
    #         est = tf.clip_by_value(est, 0.001, max_d)
    #
    #         if not new_traj:
    #             self.compiled_metrics.update_state(gt, est)
    #
    #     # Return a dict mapping metric names to current value.
    #     out_dict = {m.name: m.result() for m in self.metrics}
    #     return out_dict

    # @tf.function
    # def predict_step(self, data):
    #     # expects one sequence element at a time (batch dim is required a
    #     # nd is free to be set)"
    #     preds = self.full_model([[data], data["camera"]], training=False)
    #
    #     est = preds
    #
    #     return_data = {
    #         "image": data["RGB_im"],
    #         "depth": est["depth"],
    #         "new_traj": data["new_traj"]
    #     }
    #     return return_data

    @tf.function
    def m4depth_loss(self, gts, preds):
        with tf.name_scope("loss_function"):

            # Clip and convert depth
            def preprocess(input):
                return tf.math.log(tf.clip_by_value(input, 0.01, 200.))

            l1_loss = 0.
            for gt, pred_pyr in zip(gts[1:],
                                    preds[1:]):  # Iterate over sequence
                nbre_points = 0.

                gt_preprocessed = preprocess(gt["depth"])

                def masked_reduce_mean(array, mask, axis=None):
                    return tf.reduce_sum(array * mask, axis=axis) / (
                            tf.reduce_sum(mask, axis=axis) + 1e-12)

                for i, pred in enumerate(
                        pred_pyr):  # Iterate over the outputs produced by the different levels


                    pred_depth = preprocess(pred["depth"])

                    # Compute loss term
                    b, h, w = pred_depth.get_shape().as_list()[:3]
                    nbre_points += h * w

                    gt_resized = tf.image.resize(gt_preprocessed, [h, w])

                    l1_loss_term = (0.64 / (
                            2. ** (i - 1))) * tf.reduce_mean(
                        tf.abs(gt_resized - pred_depth))

                    l1_loss += l1_loss_term / (float(len(gts) - 1))

            return l1_loss

    def save_h5(self, name):
        self.full_model.save(
            name,
            save_format="h5",
        )


if __name__ == '__main__':
    level=2

    model = M4Depth(n_levels=level)
    print(model.full_model.summary())
    model.save_h5("m4depth_model_L_"+str(level)+".h5")
    # model.full_model.summary()
    # for i in model.full_model.layers:
    #     if "lambda" in i.name or "Lambda" in i.name:
    #         print("name {}".format(i.name))
    #         print("in {}".format(i.input))
    #         print("out {}".format(i.output))

