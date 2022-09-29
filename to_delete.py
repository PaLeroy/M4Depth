@tf.function
def get_disparity_sweeping_cv(c1, c2, disp_prev_t, disp, rot, trans, camera,
                              search_range, nbre_cuts=1):
    """ Computes the DSCV as presented in the paper """

    # Prepare inputs
    nbre_copies = 2 * search_range + 1

    expl_range = ks.layers.Reshape(
        tf.range(-search_range, search_range + 1, 1.0, dtype=tf.float32),
        [-1, 1, 1, 1, 1])

    b, h, w = c1.get_shape().as_list()[0:3]

    disp = tile_in_batch(disp, nbre_copies)
    print("disp_tile", disp.shape)

    disp = ks.layers.Reshape(disp, [nbre_copies, -1, w, h, 1])
    disp = ks.layers.Reshape(disp + expl_range,
                      [-1, h, w, 1])  # [nbre_copies*b,h,w,1]
    disp = tf.clip_by_value(disp, 1e-6, 1e6)

    # Compute disp independent factors
    coords2d, _ = get_coords_2d(c1, camera)
    coords2d = ks.layers.Reshape(coords2d, [b, h * w, 3, 1])

    rot_mat = tf.expand_dims(get_rot_mat(rot), axis=1)
    t = ks.layers.Reshape(trans, [b, 1, 3, 1])
    f_vec = ks.layers.Reshape(tf.concat([camera["f"], tf.ones([b, 1])], axis=1),
                       [b, 1, 3, 1])

    rot_coords = rot_mat @ coords2d
    alpha = rot_coords[:, :, -1:, :]
    proj_coords = rot_coords * f_vec / alpha
    scaled_t = t * f_vec

    delta_x = scaled_t[:, :, 0, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                            :, 0, 0]
    delta_y = scaled_t[:, :, 1, 0] - scaled_t[:, :, 2, 0] * proj_coords[:,
                                                            :, 1, 0]
    delta_x = ks.layers.Reshape(delta_x, [1, b, h, w, 1])
    delta_y = ks.layers.Reshape(delta_y, [1, b, h, w, 1])

    start_coords = ks.layers.Reshape(coords2d[:, :, :2, :] * f_vec[:, :, :2, :],
                              [1, b, h, w, 2])
    proj_coords = ks.layers.Reshape(proj_coords[:, :, :2, :], [1, b, h, w, 2])

    # disp to flow
    disp = ks.layers.Reshape(disp, [nbre_copies, b, h, w, 1])
    sqrt_value = tf.sqrt(delta_x ** 2 + delta_y ** 2)
    divider = sqrt_value / disp  # is correct computation after simplification
    delta = tf.concat([delta_x / divider, delta_y / divider], axis=-1)
    flow = proj_coords + delta - start_coords
    flow = ks.layers.Reshape(tf.reverse(flow, axis=[-1]),
                      [nbre_copies * b, h, w, 2])

    c1 = tile_in_batch(c1, nbre_copies)
    combined_data = tile_in_batch(tf.concat([c2, disp_prev_t], axis=-1),
                                  nbre_copies)

    combined_data_w = dense_image_warp(combined_data, flow)

    c2_w = combined_data_w[:, :, :, :-1]
    prev_disp = combined_data_w[:, :, :, -1]

    # Compute costs (operations performed in float16 for speedup)
    sub_costs = tf.stack(
        tf.split(tf.cast(c1, tf.float16) * tf.cast(c2_w, tf.float16),
                 num_or_size_splits=nbre_cuts, axis=-1), 0)
    cv = tf.reduce_mean(sub_costs, axis=-1)
    cv = tf.cast(
        tf.transpose(ks.layers.Reshape(cv, [(nbre_cuts) * nbre_copies, -1, h, w]),
                     perm=[1, 2, 3, 0]), tf.float32)

    prev_disp = tf.transpose(
        ks.layers.Reshape(prev_disp, [nbre_copies, -1, h, w]), perm=[1, 2, 3, 0])
    return cv, prev_disp
