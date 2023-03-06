"""
----------------------------------------------------------------------------------------
Copyright (c) 2022 - Michael Fonder, University of Liège (ULiège), Belgium.

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License along with this
program. If not, see < [ https://www.gnu.org/licenses/ | https://www.gnu.org/licenses/ ] >.
----------------------------------------------------------------------------------------

The functions in this file implement the equations presented in the paper. Please refer to it
for details about the operations performed here.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
from utils.dense_image_warp_functional import dense_image_warp


# def wrap_feature_block(feature_block, opt_flow):
#     with tf.compat.v1.name_scope("wrap_feature_block"):
#         feature_block = tf.identity(feature_block)
#         height, width, in_channels = feature_block.get_shape().as_list()[1:4]
#         flow = tf.image.resize_bilinear(opt_flow, [height, width])
#         scaled_flow = tf.multiply(flow, [float(height), float(width)])
#         return dense_image_warp(feature_block, scaled_flow)


def get_rot_mat(rot):
    """ Converts a rotation vector into a rotation matrix

    If the vector is of length 3 an "xyz"  small rotation sequence is expected
    If the vector is of length 4 an "wxyz" quaternion is expected

    For Midair, length is always 4.
    """
    b, c = rot.get_shape().as_list()
    if c == 3:
        # We never go here normally
        ones = tf.ones([b])
        matrix = tf.stack((ones, -rot[:, 2], rot[:, 1],
                           rot[:, 2], ones, -rot[:, 0],
                           -rot[:, 1], rot[:, 0], ones), axis=-1)

        output_shape = tf.concat((tf.shape(input=rot)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)
    elif c == 4:
        w, x, y, z = tf.unstack(rot, axis=-1)
        tx = 2.0 * x
        ty = 2.0 * y
        tz = 2.0 * z
        twx = tx * w
        twy = ty * w
        twz = tz * w
        txx = tx * x
        txy = ty * x
        txz = tz * x
        tyy = ty * y
        tyz = tz * y
        tzz = tz * z
        matrix = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                           txy + twz, 1.0 - (txx + tzz), tyz - twx,
                           txz - twy, tyz + twx, 1.0 - (txx + tyy)),
                          axis=-1)  # pyformat: disable
        output_shape = tf.concat((tf.shape(input=rot)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)
    else:
        raise ValueError(
            'Rotation must be expressed as a small angle (x,y,z) or a quaternion (w,x,y,z)')

@tf.function
def repeat_ones(tensor):
    myconst = tf.convert_to_tensor(np.ones((1, 1)).astype('float32'))
    shapes = tf.shape(tensor)
    return tf.repeat(myconst, shapes[0], axis=0)



@tf.function
def repeat_const(tensor, myconst):
    shapes = tf.shape(tensor)
    return tf.repeat(myconst, shapes[0], axis=0)

@tf.function
def get_coords_2d(map, camera_c, camera_f):
    """ Creates a grid containing pixel coordinates normalized by the camera focal length """
    # Modif test
    h, w = map.get_shape().as_list()[1:3]
    h_range = tf.range(0., h, 1.0, dtype=tf.float32) + 0.5
    w_range = tf.range(0., w, 1.0, dtype=tf.float32) + 0.5

    new_cam_c = ks.layers.Reshape((1, 1, 2,), )(camera_c)
    new_cam_f = ks.layers.Reshape((1, 1, 2,), )(camera_f)

    grid_x, grid_y = tf.meshgrid(w_range, h_range)
    mesh = tf.reshape(tf.stack([grid_x, grid_y], axis=2), [1, h, w, 2])

    mesh = mesh - new_cam_c
    divide = tf.divide(mesh, new_cam_f)

    myconst = tf.convert_to_tensor(np.ones((1, h, w, 1)).astype('float32'))
    ones_ = tf.keras.layers.Lambda(lambda x: repeat_const(x, myconst))(map)
    # Petit rappel pour te dire que tu as mis 1 journée à trouver ça

    coords_2d = ks.layers.Concatenate()([divide, ones_])
    coords_2d = tf.expand_dims(coords_2d, -1)
    return coords_2d, mesh

@tf.function
def get_coords_2d_former(map, camera):
    """ Creates a grid containing pixel coordinates normalized by the camera focal length """
    b, h, w, c = map.get_shape().as_list()
    h_range = tf.range(0., h, 1.0, dtype=tf.float32) + 0.5
    w_range = tf.range(0., w, 1.0, dtype=tf.float32) + 0.5
    grid_x, grid_y = tf.meshgrid(w_range, h_range)
    mesh = tf.reshape(tf.stack([grid_x, grid_y], axis=2), [1, h, w, 2]) - tf.reshape(camera["c"], [b, 1, 1, 2])

    divide = tf.divide(mesh, tf.reshape(camera["f"], [b, 1, 1, 2]))

    coords_2d = tf.concat([divide, tf.ones([b,h,w,1])], axis=-1)
    coords_2d = tf.expand_dims(coords_2d, -1)
    return coords_2d, mesh

@tf.function
def reproject(map, depth, rot, trans, camera):
    """ Spatially warps (reprojects) an input feature map according to given depth map, motion and camera properties """

    with tf.name_scope("reproject"):
        # Test the shape of the inputs
        b, h, w, c = map.get_shape().as_list()
        b, h1, w1, c = depth.get_shape().as_list()
        if w != w1 or h != h1:
            raise ValueError(
                'Height and width of map and depth should be the same')

        # Reshape motion data in a format compatible for vector math
        fx = camera["f"][:, 0]
        fy = camera["f"][:, 1]

        proj_mat = []
        for i in range(b):
            proj_mat.append([[fx[i], 0., 0.], [0., fy[i], 0.], [0., 0., 1.]])
        proj_mat = tf.convert_to_tensor(proj_mat)

        rot_mat = get_rot_mat(rot)
        transformation_mat = tf.concat([rot_mat, tf.expand_dims(trans, -1)],
                                       -1)

        # Fuse projection matrix K with transformation matrix
        combined_mat = tf.linalg.matmul(proj_mat, transformation_mat)
        combined_mat = tf.reshape(combined_mat, [b, 1, 1, 3, 4])

        # Get the relative coordinates for each point of the map
        coords, mesh = get_coords_2d(map, camera["c"], camera["f"])
        pos_vec = tf.expand_dims(
            tf.concat([coords[:, :, :, :, 0] * depth, tf.ones([b, h, w, 1])],
                      axis=-1), axis=-1)

        # Compute corresponding coordinates in related frame
        proj_pos = tf.linalg.matmul(combined_mat, pos_vec)
        proj_coord = proj_pos[:, :, :, :2, 0] / proj_pos[:, :, :, 2:, 0]
        rot_pos = tf.linalg.matmul(combined_mat[:, :, :, :, :3],
                                   pos_vec[:, :, :, :3, :])
        rot_coord = rot_pos[:, :, :, :2, 0] / rot_pos[:, :, :, 2:, 0]

        flow = tf.reverse(proj_coord - mesh, axis=[-1])

    return dense_image_warp(map, flow), [proj_coord - rot_coord, rot_coord]


@tf.function
def recompute_depth(depth, rot, trans, camera, mesh=None):
    """ Recomputes perceived depth according to given camera motion and specifications """

    with tf.compat.v1.name_scope("recompute_depth"):
        depth = tf.identity(depth)
        b, h, w, c = depth.get_shape().as_list()

        # Reshape motion data in a format compatible for vector math
        trans_vec = tf.reshape(-trans, [b, 1, 1, 3, 1])
        rot_mat = get_rot_mat(rot)[:, -1:, :]

        # Get the relative coordinates for each point of the map
        if mesh is None:
            h_range = tf.range(0., h, 1.0, dtype=tf.float32) + 0.5
            w_range = tf.range(0., w, 1.0, dtype=tf.float32) + 0.5
            grid_x, grid_y = tf.meshgrid(w_range, h_range)
            mesh = tf.reshape(tf.stack([grid_x, grid_y], axis=2),
                              [1, h, w, 2]) - tf.reshape(camera["c"],
                                                         [b, 1, 1, 2])

        coords_2d = tf.concat(
            [tf.divide(mesh, tf.reshape(camera["f"], [b, 1, 1, 2])),
             tf.ones([b, h, w, 1])], axis=-1)
        pos_vec = tf.expand_dims(coords_2d, -1)

        # Recompute depth
        trans_vec = tf.linalg.matmul(tf.reshape(rot_mat, [b, 1, 1, 1, 3]),
                                     trans_vec)
        proj_pos_rel = tf.linalg.matmul(tf.reshape(rot_mat, [b, 1, 1, 1, 3]),
                                        pos_vec)
        new_depth = tf.stop_gradient(
            proj_pos_rel[:, :, :, :, 0]) * depth + tf.stop_gradient(
            trans_vec[:, :, :, :, 0])
        return tf.clip_by_value(new_depth, 0.1, 2000.)

@tf.function
def lambda_disp2depth(disp, rot, trans, camera_c, camera_f):
    """ Converts a disparity map into a depth map according to given camera motion and specifications """
    b, h, w = disp.shape[0:3]
    coords2d, _ = get_coords_2d(disp, camera_c, camera_f)
    disp = ks.layers.Reshape((h * w, 1, 1,), )(disp)

    max_function = lambda x : tf.maximum(x, 1e-5)
    disp = ks.layers.Lambda(max_function)(disp)

    coords2d = ks.layers.Reshape((h * w, 3, 1,), )(coords2d)
    rot_mat = get_rot_mat(rot)
    # rot_mat = tf.expand_dims(rot_mat, axis=1)
    # expand_function = lambda x : tf.expand_dims(x, axis=1)
    # rot_mat = ks.layers.Lambda(expand_function)(rot_mat)
    rot_mat = ks.layers.Reshape((1, rot_mat.shape[1], rot_mat.shape[2],), )(rot_mat)

    t = ks.layers.Reshape((1, 3, 1,), )(trans)
    myconst = tf.convert_to_tensor(np.ones((1, 1)).astype('float32'))
    ones_ = tf.keras.layers.Lambda(lambda x: repeat_const(x, myconst))(camera_f)
    f_vec = ks.layers.Concatenate(axis=1)([camera_f, ones_])
    f_vec =  ks.layers.Reshape((1, 3, 1,),)(f_vec)

    rot_coords = rot_mat @ coords2d
    alpha = rot_coords[:, :, -1:, :]
    proj_coords = rot_coords * f_vec / alpha
    scaled_t = t * f_vec

    delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                            :, 0, 0]
    delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                            :, 1, 0]

    sqrt_value = tf.sqrt(delta_x ** 2 + delta_y ** 2)
    sqrt_value = ks.layers.Reshape((h * w, 1, 1,), )(sqrt_value)


    depth = (sqrt_value / disp - scaled_t[:, :, -1:, :]) / alpha
    depth = ks.layers.Reshape((h, w, 1,), )(depth)

    return depth


def disp2depth(disp, rot, trans, camera_c, camera_f):
    """ Converts a disparity map into a depth map according to given camera motion and specifications """
    b, h, w = disp.shape[0:3]
    get_coords_2d_lambda = lambda x: get_coords_2d(x[0], x[1], x[2])
    coords2d, _ = ks.layers.Lambda(get_coords_2d_lambda)((disp, camera_c, camera_f))
    disp = ks.layers.Reshape((h * w, 1, 1,), )(disp)

    max_function = lambda x: tf.maximum(x, 1e-5)
    disp = ks.layers.Lambda(max_function)(disp)

    coords2d = ks.layers.Reshape((h * w, 3, 1,), )(coords2d)
    rot_mat = get_rot_mat(rot)
    # rot_mat = tf.expand_dims(rot_mat, axis=1)
    # expand_function = lambda x : tf.expand_dims(x, axis=1)
    # rot_mat = ks.layers.Lambda(expand_function)(rot_mat)
    rot_mat = ks.layers.Reshape((1, rot_mat.shape[1], rot_mat.shape[2],), )(rot_mat)
    t = ks.layers.Reshape((1, 3, 1,), )(trans)

    ones_ = tf.keras.layers.Lambda(lambda x: repeat_ones(x))(camera_f)
    f_vec = ks.layers.Concatenate(axis=1)([camera_f, ones_])
    # TODO tf.pad
    f_vec =  ks.layers.Reshape((1, 3, 1,),)(f_vec)

    rot_coords = rot_mat @ coords2d
    alpha = rot_coords[:, :, -1:, :]
    proj_coords = rot_coords * f_vec / alpha
    scaled_t = t * f_vec

    delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                            :, 0, 0]
    delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                            :, 1, 0]

    sqrt_value = tf.sqrt(delta_x ** 2 + delta_y ** 2)
    sqrt_value = ks.layers.Reshape((h * w, 1, 1,), )(sqrt_value)


    depth = (sqrt_value / disp - scaled_t[:, :, -1:, :]) / alpha
    depth = ks.layers.Reshape((h, w, 1,), )(depth)

    return depth

@tf.function
def disp2depth_former(disp, rot, trans, camera):
    """ Converts a disparity map into a depth map according to given camera motion and specifications """

    with tf.compat.v1.name_scope("disp2depth"):
        b, h, w = disp.get_shape().as_list()[0:3]

        coords2d, _ = get_coords_2d(disp, camera)
        disp = tf.maximum(tf.reshape(disp, [b, h * w, 1, 1]), 1e-5)
        coords2d = tf.reshape(coords2d, [b, h * w, 3, 1])
        rot_mat = tf.expand_dims(get_rot_mat(rot), axis=1)
        t = tf.reshape(trans, [b, 1, 3, 1])
        f_vec = tf.reshape(tf.concat([camera["f"], tf.ones([b, 1])], axis=1),
                           [b, 1, 3, 1])

        rot_coords = rot_mat @ coords2d
        alpha = rot_coords[:, :, -1:, :]
        proj_coords = rot_coords * f_vec / alpha
        scaled_t = t * f_vec

        delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                                :, 0, 0]
        delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                                :, 1, 0]

        sqrt_value = tf.reshape(tf.sqrt(delta_x ** 2 + delta_y ** 2),
                                [b, h * w, 1, 1])

        depth = (sqrt_value / disp - scaled_t[:, :, -1:, :]) / alpha

        to_ret= tf.reshape(depth, [b, h, w, 1])

        return to_ret

@tf.function
def depth2disp(depth, rot, trans, camera_c, camera_f):
    """ Converts a depth map into a disparity map according to given camera motion and specifications """

    b, h, w = depth.get_shape().as_list()[0:3]

    coords2d, _ = get_coords_2d(depth, camera_c, camera_f)

    depth = ks.layers.Reshape((h * w, 1, 1,), )(depth)
    coords2d = ks.layers.Reshape((h * w, 3, 1,), )(coords2d)

    rot_mat = get_rot_mat(rot)
    rot_mat = ks.layers.Reshape((1, rot_mat.shape[1], rot_mat.shape[2],), )(rot_mat)
    t = ks.layers.Reshape((1, 3, 1,), )(trans)

    myconst = tf.convert_to_tensor(np.ones((1, 1)).astype('float32'))
    ones_ = tf.keras.layers.Lambda(lambda x: repeat_const(x, myconst))(camera_f)
    f_vec = ks.layers.Concatenate(axis=1)([camera_f, ones_])
    f_vec = ks.layers.Reshape((1, 3, 1,),)(f_vec)

    rot_coords = rot_mat @ coords2d
    alpha = rot_coords[:, :, -1:, :]
    proj_coords = rot_coords * f_vec / alpha
    scaled_t = t * f_vec

    delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                            :, 0, 0]
    delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                            :, 1, 0]
    sqrt_value = tf.sqrt(delta_x ** 2 + delta_y ** 2)
    sqrt_value = ks.layers.Reshape((h * w, 1, 1,), ) (sqrt_value)

    disp = sqrt_value / (depth * alpha + scaled_t[:, :, -1:, :])

    disp = ks.layers.Reshape((h, w, 1,), ) (disp)

    return disp

@tf.function
def depth2disp_former(depth, rot, trans, camera):
    """ Converts a depth map into a disparity map according to given camera motion and specifications """

    with tf.compat.v1.name_scope("depth2disp"):
        b, h, w = depth.get_shape().as_list()[0:3]

        coords2d, _ = get_coords_2d(depth, camera)

        depth = tf.reshape(depth, [b, h * w, 1, 1])
        coords2d = tf.reshape(coords2d, [b, h * w, 3, 1])
        rot_mat = tf.expand_dims(get_rot_mat(rot), axis=1)
        t = tf.reshape(trans, [b, 1, 3, 1])
        f_vec = tf.reshape(tf.concat([camera["f"], tf.ones([b, 1])], axis=1),
                           [b, 1, 3, 1])

        rot_coords = rot_mat @ coords2d
        alpha = rot_coords[:, :, -1:, :]
        proj_coords = rot_coords * f_vec / alpha
        scaled_t = t * f_vec

        delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                                :, 0, 0]
        delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                                :, 1, 0]

        sqrt_value = tf.reshape(tf.sqrt(delta_x ** 2 + delta_y ** 2),
                                [b, h * w, 1, 1])

        disp = sqrt_value / (depth * alpha + scaled_t[:, :, -1:, :])

        return tf.reshape(disp, [b, h, w, 1])

def prev_d2disp(prev_d, rot, trans, camera_c, camera_f):
    """ Converts depth map corresponding to previous time step into the disparity map corresponding to current time step """
    b, h, w = prev_d.get_shape().as_list()[0:3]

    get_coords_2d_lambda = lambda x: get_coords_2d(x[0], x[1], x[2])
    coords2d, _ = ks.layers.Lambda(get_coords_2d_lambda)((prev_d, camera_c, camera_f))
    prev_d = ks.layers.Reshape([h * w, 1, 1, ],)(prev_d)
    coords2d = ks.layers.Reshape([h * w, 3, 1, ], )(coords2d)

    t = ks.layers.Reshape([1, 3, 1, ],)(trans)
    ones_ = tf.keras.layers.Lambda(lambda x: repeat_ones(x))(camera_f)
    f_vec = ks.layers.Concatenate(axis=1)([camera_f, ones_])
    f_vec = ks.layers.Reshape((1, 3, 1,),)(f_vec)

    coords2d = coords2d * f_vec
    scaled_t = t * f_vec

    # delta = (scaled_t - t[:, :, -1:, :] * coords2d) / (
    #             prev_d - t[:, :, -1:, :])
    delta = tf.math.divide_no_nan(scaled_t - t[:, :, -1:, :] * coords2d, prev_d - t[:, :, -1:, :])
    disp = tf.norm(delta[:, :, :2, :], axis=2)
    disp = ks.layers.Reshape([h, w, 1,], )(disp)
    disp = tf.stop_gradient(disp)
    return disp

@tf.function
def lambda_prev_d2disp(prev_d, rot, trans, camera_c, camera_f):
    """ Converts depth map corresponding to previous time step into the disparity map corresponding to current time step """
    b, h, w = prev_d.get_shape().as_list()[0:3]
    count = tf.math.count_nonzero(tf.math.is_nan(prev_d))
    if count != 0:
        tf.print("nan found prev_d args 1")
        print("nan found prev_d args 1")

    coords2d, _ = get_coords_2d(prev_d, camera_c, camera_f)
    prev_d = ks.layers.Reshape([h * w, 1, 1, ],)(prev_d)
    coords2d = ks.layers.Reshape([h * w, 3, 1, ], )(coords2d)

    count = tf.math.count_nonzero(tf.math.is_nan(prev_d))
    if count != 0:
        tf.print("nan found prev_d reshaped 2")
        print("nan found prev_d reshaped 2")

    t = ks.layers.Reshape([1, 3, 1, ],)(trans)
    myconst = tf.convert_to_tensor(np.ones((1, 1)).astype('float32'))
    ones_ = tf.keras.layers.Lambda(lambda x: repeat_const(x, myconst))(camera_f)
    f_vec = ks.layers.Concatenate(axis=1)([camera_f, ones_])
    f_vec = ks.layers.Reshape((1, 3, 1,),)(f_vec)

    coords2d = coords2d * f_vec
    scaled_t = t * f_vec

    count = tf.math.count_nonzero(tf.math.is_nan(coords2d))
    if count != 0:
        tf.print("nan found coords2d")
        print("nan found coords2d")

    count = tf.math.count_nonzero(tf.math.is_nan(scaled_t))
    if count != 0:
        tf.print("nan found scaled_t")
        print("nan found scaled_t")

    count = tf.math.count_nonzero(tf.math.is_nan(prev_d))
    if count != 0:
        tf.print("nan found prev_d")
        print("nan found prev_d")

    # delta = (scaled_t - t[:, :, -1:, :] * coords2d) / (
    #             prev_d - t[:, :, -1:, :])

    delta = tf.math.divide_no_nan(scaled_t - t[:, :, -1:, :] * coords2d, prev_d - t[:, :, -1:, :])

    count = tf.math.count_nonzero(tf.math.is_nan(delta))
    if count != 0:
        tf.print("nan found delta")
        print("nan found delta")

    disp = tf.norm(delta[:, :, :2, :], axis=2)
    disp = ks.layers.Reshape([h, w, 1,], )(disp)
    disp = tf.stop_gradient(disp)
    count = tf.math.count_nonzero(tf.math.is_nan(disp))
    if count != 0:
        tf.print("nan found disp")
        print("nan found disp")
    return disp

@tf.function
def prev_d2disp_former(prev_d, rot, trans, camera):
    """ Converts depth map corresponding to previous time step into the disparity map corresponding to current time step """

    with tf.compat.v1.name_scope("prev_d2disp"):
        b, h, w = prev_d.get_shape().as_list()[0:3]

        coords2d, _ = get_coords_2d(prev_d, camera)

        prev_d = tf.reshape(prev_d, [b, h * w, 1, 1])
        coords2d = tf.reshape(coords2d, [b, h * w, 3, 1])
        t = tf.reshape(trans, [b, 1, 3, 1])
        f_vec = tf.reshape(tf.concat([camera["f"], tf.ones([b, 1])], axis=1),
                           [b, 1, 3, 1])

        coords2d = coords2d * f_vec
        scaled_t = t * f_vec

        delta = (scaled_t - t[:, :, -1:, :] * coords2d) / (
                    prev_d - t[:, :, -1:, :])

        disp = tf.norm(delta[:, :, :2, :], axis=2)

        return tf.stop_gradient(tf.reshape(disp, [b, h, w, 1]))


def tile_not_in_batch(map, nbre_copies):
    """
    tile_not_in_batch returns
    (b,nbre_copies, ...)
    instead of tile_in_batch that returns
    (b*nbre_copies,....)
    tile_in_batch[1] = tile_in_batch_batchsize[1][0]
    """
    map_shape = map.get_shape().as_list()
    map = tf.expand_dims(map, axis=1)

    map = tf.tile(map, [1] + [nbre_copies] + [1 for i in map_shape[1:]])

    return map

def get_disparity_sweeping_cv(inp, search_range, nbre_cuts=1):
    """ Computes the DSCV as presented in the paper """
    c1, c2, disp_prev_t, disp, rot, trans, camera_c,  camera_f \
        = inp[0], inp[1], inp[2], inp[3], inp[4], inp[5], inp[6], inp[7]

    # Prepare inputs
    nbre_copies = 2 * search_range + 1
    range_before_reshape = tf.range(-search_range, search_range + 1, 1.0, dtype=tf.float32)
    expl_range = tf.reshape(range_before_reshape , [1, -1, 1, 1, 1])
    b, h, w = c1.get_shape().as_list()[0:3]

    disp = tile_not_in_batch(disp, nbre_copies)
    disp = tf.reshape(disp, [-1, nbre_copies, w, h, 1])

    disp = disp + expl_range
    disp = tf.clip_by_value(disp, 1e-6, 1e6)
    # Compute disp independent factors
    get_coords_2d_lambda = lambda x: get_coords_2d(x[0], x[1], x[2])
    coords2d, _ = ks.layers.Lambda(get_coords_2d_lambda)((c1, camera_c, camera_f))
    coords2d = ks.layers.Reshape([h * w, 3, 1, ], )(coords2d)
    # rot_mat = tf.expand_dims(get_rot_mat(rot), axis=1)
    rot_mat = get_rot_mat(rot)
    rot_mat = ks.layers.Reshape((1, rot_mat.shape[1], rot_mat.shape[2],), )(rot_mat)
    t = ks.layers.Reshape((1, 3, 1, ), )(trans)

    ones_ = tf.keras.layers.Lambda(lambda x: repeat_ones(x))(camera_f)
    f_vec = ks.layers.Concatenate(axis=1)([camera_f, ones_])
    f_vec =  ks.layers.Reshape((1, 3, 1,),)(f_vec)
    rot_coords = rot_mat @ coords2d
    alpha = rot_coords[:, :, -1:, :]
    proj_coords = rot_coords * f_vec / alpha
    scaled_t = t * f_vec
    delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                            :, 0, 0]
    delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                            :, 1, 0]
    delta_x = ks.layers.Reshape([1, h, w, 1,],)(delta_x)
    delta_y = ks.layers.Reshape([1, h, w, 1,],)(delta_y)

    start_coords = ks.layers.Reshape([ 1, h, w, 2,],)(coords2d[:, :, :2, :] * f_vec[:, :, :2, :])
    proj_coords = ks.layers.Reshape([1, h, w, 2,],)(proj_coords[:, :, :2, :])

    # disp to flow
    sqrt_value = tf.sqrt(delta_x ** 2 + delta_y ** 2)
    divider = sqrt_value / disp  # is correct computation after simplification
    delta = tf.concat([delta_x / divider, delta_y / divider], axis=-1)
    flow = proj_coords + delta - start_coords
    flow = tf.reverse(flow, axis=[-1])
    c1 = tile_not_in_batch(c1, nbre_copies)
    combined_data = tile_not_in_batch(tf.concat([c2, disp_prev_t], axis=-1),
                                  nbre_copies)
    dense_image_warp_layer = ks.layers.Lambda(lambda x: dense_image_warp(x[0], x[1]))
    combined_data_w = dense_image_warp_layer((combined_data, flow))
    # combined_data_w = combined_data
    c2_w = combined_data_w[..., :-1]
    prev_disp = combined_data_w[..., -1]
    # Compute costs (operations performed in float16 for speedup)
    mult = tf.cast(c1, tf.float16) * tf.cast(c2_w, tf.float16)
    split = tf.split(mult, num_or_size_splits=nbre_cuts, axis=-1)
    sub_costs = tf.stack(split, 1)
    cv = tf.reduce_mean(sub_costs, axis=-1)
    cv_to_cast = tf.reshape(cv, [-1, (nbre_cuts) * nbre_copies, h, w])
    cv = tf.cast(tf.transpose(cv_to_cast, perm=[0, 2, 3, 1]), tf.float32)
    prev_disp = tf.transpose(prev_disp, perm=[0, 2, 3, 1])

    return cv, prev_disp


@tf.function
def lambda_get_disparity_sweeping_cv(inp, search_range, nbre_cuts=1):
    """ Computes the DSCV as presented in the paper """
    c1, c2, disp_prev_t, disp, rot, trans, camera_c,  camera_f \
        = inp[0], inp[1], inp[2], inp[3], inp[4], inp[5], inp[6], inp[7]

    # Prepare inputs
    nbre_copies = 2 * search_range + 1
    range_before_reshape = tf.range(-search_range, search_range + 1, 1.0, dtype=tf.float32)
    expl_range = tf.reshape(range_before_reshape , [1, -1, 1, 1, 1])
    b, h, w = c1.get_shape().as_list()[0:3]

    disp = tile_not_in_batch(disp, nbre_copies)
    disp = tf.reshape(disp, [-1, nbre_copies, w, h, 1])

    disp = disp + expl_range
    disp = tf.clip_by_value(disp, 1e-6, 1e6)
    # Compute disp independent factors
    coords2d, _ = get_coords_2d(c1, camera_c, camera_f)
    coords2d = ks.layers.Reshape([h * w, 3, 1,])(coords2d)
    # rot_mat = tf.expand_dims(get_rot_mat(rot), axis=1)
    rot_mat = get_rot_mat(rot)
    rot_mat = ks.layers.Reshape((1, rot_mat.shape[1], rot_mat.shape[2],), )(rot_mat)
    t = ks.layers.Reshape((1, 3, 1, ), )(trans)

    myconst = tf.convert_to_tensor(np.ones((1, 1)).astype('float32'))
    ones_ = tf.keras.layers.Lambda(lambda x: repeat_const(x, myconst))(camera_f)
    f_vec = ks.layers.Concatenate(axis=1)([camera_f, ones_])
    f_vec =  ks.layers.Reshape((1, 3, 1,),)(f_vec)
    rot_coords = rot_mat @ coords2d
    alpha = rot_coords[:, :, -1:, :]
    proj_coords = rot_coords * f_vec / alpha
    scaled_t = t * f_vec
    delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                            :, 0, 0]
    delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                            :, 1, 0]
    delta_x = ks.layers.Reshape([1, h, w, 1,],)(delta_x)
    delta_y = ks.layers.Reshape([1, h, w, 1,],)(delta_y)

    start_coords = ks.layers.Reshape([ 1, h, w, 2,],)(coords2d[:, :, :2, :] * f_vec[:, :, :2, :])
    proj_coords = ks.layers.Reshape([1, h, w, 2,],)(proj_coords[:, :, :2, :])

    # disp to flow
    sqrt_value = tf.sqrt(delta_x ** 2 + delta_y ** 2)
    divider = sqrt_value / disp  # is correct computation after simplification
    delta = tf.concat([delta_x / divider, delta_y / divider], axis=-1)
    flow = proj_coords + delta - start_coords
    flow = tf.reverse(flow, axis=[-1])
    c1 = tile_not_in_batch(c1, nbre_copies)
    combined_data = tile_not_in_batch(tf.concat([c2, disp_prev_t], axis=-1),
                                  nbre_copies)
    combined_data_w = dense_image_warp(combined_data, flow)
    # combined_data_w = combined_data
    c2_w = combined_data_w[..., :-1]
    prev_disp = combined_data_w[..., -1]
    # Compute costs (operations performed in float16 for speedup)
    mult = tf.cast(c1, tf.float16) * tf.cast(c2_w, tf.float16)
    split = tf.split(mult, num_or_size_splits=nbre_cuts, axis=-1)
    sub_costs = tf.stack(split, 1)
    cv = tf.reduce_mean(sub_costs, axis=-1)
    cv_to_cast = tf.reshape(cv, [-1, (nbre_cuts) * nbre_copies, h, w])
    cv = tf.cast(tf.transpose(cv_to_cast, perm=[0, 2, 3, 1]), tf.float32)
    prev_disp = tf.transpose(prev_disp, perm=[0, 2, 3, 1])

    return cv, prev_disp

@tf.function
def get_disparity_sweeping_cv_former_intermediate(c1, c2, disp_prev_t, disp, rot, trans, camera,
                              search_range, nbre_cuts=1):
    """ Computes the DSCV as presented in the paper """

    with tf.compat.v1.name_scope("DSCV"):
        # Prepare inputs
        nbre_copies = 2 * search_range + 1
        range_before_reshape = tf.range(-search_range, search_range + 1, 1.0, dtype=tf.float32)
        expl_range = tf.reshape(range_before_reshape , [1, -1, 1, 1, 1])
        b, h, w = c1.get_shape().as_list()[0:3]

        disp = tile_not_in_batch(disp, nbre_copies)
        disp = tf.reshape(disp, [-1, nbre_copies, w, h, 1])

        disp = disp + expl_range
        disp = tf.clip_by_value(disp, 1e-6, 1e6)
        # Compute disp independent factors
        coords2d, _ = get_coords_2d(c1, camera)
        coords2d = tf.reshape(coords2d, [b, h * w, 3, 1])
        rot_mat = tf.expand_dims(get_rot_mat(rot), axis=1)
        t = tf.reshape(trans, [b, 1, 3, 1])
        f_vec = tf.reshape(tf.concat([camera["f"], tf.ones([b, 1])], axis=1),
                           [b, 1, 3, 1])

        rot_coords = rot_mat @ coords2d
        alpha = rot_coords[:, :, -1:, :]
        proj_coords = rot_coords * f_vec / alpha
        scaled_t = t * f_vec

        delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                                :, 0, 0]
        delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                                :, 1, 0]
        delta_x = tf.reshape(delta_x, [b, 1, h, w, 1])
        delta_y = tf.reshape(delta_y, [b, 1, h, w, 1])

        start_coords = tf.reshape(coords2d[:, :, :2, :] * f_vec[:, :, :2, :],
                                  [b, 1, h, w, 2])
        proj_coords = tf.reshape(proj_coords[:, :, :2, :], [b, 1, h, w, 2])

        # disp to flow
        sqrt_value = tf.sqrt(delta_x ** 2 + delta_y ** 2)
        divider = sqrt_value / disp  # is correct computation after simplification

        delta = tf.concat([delta_x / divider, delta_y / divider], axis=-1)
        flow = proj_coords + delta - start_coords
        flow = tf.reverse(flow, axis=[-1])

        c1 = tile_not_in_batch(c1, nbre_copies)
        combined_data = tile_not_in_batch(tf.concat([c2, disp_prev_t], axis=-1),
                                      nbre_copies)


        combined_data_w = dense_image_warp(combined_data, flow)
        # combined_data_w = combined_data

        c2_w = combined_data_w[..., :-1]
        prev_disp = combined_data_w[..., -1]
        # Compute costs (operations performed in float16 for speedup)
        mult = tf.cast(c1, tf.float16) * tf.cast(c2_w, tf.float16)
        split = tf.split(mult, num_or_size_splits=nbre_cuts, axis=-1)
        sub_costs = tf.stack(split, 1)
        cv = tf.reduce_mean(sub_costs, axis=-1)
        cv_to_cast = tf.reshape(cv, [-1, (nbre_cuts) * nbre_copies, h, w])

        # cv_to_cast (18, 5, 48, 48)
        # cv_cast (5, 48, 48, 18)

        cv = tf.cast(tf.transpose(cv_to_cast, perm=[0, 2, 3, 1]), tf.float32)
        tmp = prev_disp
        prev_disp = tf.transpose(prev_disp , perm=[0, 2, 3, 1])
        return cv, prev_disp
@tf.function
def get_disparity_sweeping_cv_former(c1, c2, disp_prev_t, disp, rot, trans, camera,
                              search_range, nbre_cuts=1):
    from utils.depth_operations import get_disparity_sweeping_cv
    a,b= get_disparity_sweeping_cv(c1, c2, disp_prev_t, disp, rot, trans, camera, search_range, nbre_cuts)
    return a, b

# def cost_volume(c1, search_range, name="cost_volume", dilation_rate=1,
#                 nbre_cuts=1):
#     """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
#     Args:
#         c1: Feature map 1
#         c2: Feature map 2
#         search_range: Search range (maximum displacement)
#     """
#     c1 = tf.cast(c1, tf.float16)
#     c2 = c1
#     strided_search_range = search_range * dilation_rate
#     padded_lvl = tf.pad(c2, [[0, 0],
#                              [strided_search_range, strided_search_range],
#                              [strided_search_range, strided_search_range],
#                              [0, 0]])
#     _, h, w, _ = c2.get_shape().as_list()
#     max_offset = search_range * 2 + 1
#     c1_nchw = tf.transpose(c1, perm=[0, 3, 1, 2])
#     pl_nchw = tf.transpose(padded_lvl, perm=[0, 3, 1, 2])
#
#     c1_nchw = tf.stack(
#         tf.split(c1_nchw, num_or_size_splits=nbre_cuts, axis=1), axis=4)
#     pl_nchw = tf.stack(
#         tf.split(pl_nchw, num_or_size_splits=nbre_cuts, axis=1), axis=4)
#
#     # GATHERND
#     cost_vol = []
#     for y in range(0, max_offset):
#         for x in range(0, max_offset):
#             slice = tf.slice(pl_nchw,
#                              [0, 0, y * dilation_rate, x * dilation_rate,0],
#                              [-1, -1, h, w, -1])
#
#             cost = tf.reduce_mean(c1_nchw * slice, axis=1)
#             cost_vol.append(cost)
#     cost_vol = tf.concat(cost_vol, axis=3)
#     cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)
#
#     return tf.cast(cost_vol, tf.float32)

def cost_volume(c1, search_range, name="cost_volume", dilation_rate=1,
                nbre_cuts=1):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Feature map 1
        c2: Feature map 2
        search_range: Search range (maximum displacement)
    """
    print("in", c1.shape)
    max_offset=search_range*2+1
    c1 = tf.cast(c1, tf.float16)

    strided_search_range = search_range * dilation_rate
    padded_lvl = tf.pad(c1, [[0, 0],
                             [strided_search_range, strided_search_range],
                             [strided_search_range, strided_search_range],
                             [0, 0]])
    _, h, w, _ = c1.get_shape().as_list()
    c1_nchw = c1 #tf.transpose(c1, perm=[0, 3, 1, 2])
    pl_nchw = padded_lvl #tf.transpose(padded_lvl, perm=[0, 3, 1, 2])

    list_all2=tf.image.extract_patches(pl_nchw,
                                       sizes=[1, h, w, 1],
                                       strides=[1, dilation_rate, dilation_rate,1],
                                       rates=[1,1,1,1],
                                       padding="VALID")
    # list_all2=tf.reshape(list_all2, list_all2.shape[0:3]+c1.shape[1:4])
    list_all2=ks.layers.Reshape(list_all2.shape[1:3]+c1.shape[1:4], dtype=tf.float16)(list_all2)

    c1_repeat = tf.expand_dims(c1,axis=1)
    c1_repeat = tf.expand_dims(c1_repeat,axis=1)
    c1_repeat = tf.tile(c1_repeat, [1,list_all2.shape[1], list_all2.shape[2],1, 1, 1])
    list_all2 = c1_repeat*list_all2

    split_stack = tf.stack(tf.split(list_all2, num_or_size_splits=nbre_cuts, axis=-1), axis=-1)
    cost = tf.reduce_mean(split_stack, axis=-2)
    cost = tf.transpose(cost, perm=[0, 3, 4, 1, 2, 5])
    # cost_vol = tf.reshape(cost, cost.shape[0:1] + cost.shape[1:3] + [cost.shape[3]*cost.shape[4]*cost.shape[5]])
    cost_vol = ks.layers.Reshape(cost.shape[1:3] + [cost.shape[3]*cost.shape[4]*cost.shape[5]], dtype=tf.float16)(cost)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

    return tf.cast(cost_vol, tf.float32)

@tf.function
def lambda_cost_volume(c1, search_range, name="cost_volume", dilation_rate=1,
                nbre_cuts=1):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Feature map 1
        c2: Feature map 2
        search_range: Search range (maximum displacement)
    """
    c2 = c1
    c1 = tf.cast(c1, tf.float16)
    c2 = tf.cast(c2, tf.float16)
    strided_search_range = search_range * dilation_rate
    padded_lvl = tf.pad(c2, [[0, 0],
                             [strided_search_range, strided_search_range],
                             [strided_search_range, strided_search_range],
                             [0, 0]])
    _, h, w, _ = c2.get_shape().as_list()
    max_offset = search_range * 2 + 1
    c1_nchw = tf.transpose(c1, perm=[0, 3, 1, 2])
    pl_nchw = tf.transpose(padded_lvl, perm=[0, 3, 1, 2])

    c1_nchw = tf.stack(
        tf.split(c1_nchw, num_or_size_splits=nbre_cuts, axis=1), axis=4)
    pl_nchw = tf.stack(
        tf.split(pl_nchw, num_or_size_splits=nbre_cuts, axis=1), axis=4)

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(pl_nchw,
                             [0, 0, y * dilation_rate, x * dilation_rate,
                              0], [-1, -1, h, w, -1])
            cost = tf.reduce_mean(c1_nchw * slice, axis=1)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.1, name=name)

    return tf.cast(cost_vol, tf.float32)
