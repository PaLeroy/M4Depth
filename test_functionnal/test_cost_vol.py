import tensorflow as tf

c1=tf.random.uniform((4, 96, 96, 32))
max_offset=7
dilation_rate=1
search_range=3
nbre_cuts=2


c1 = tf.cast(c1, tf.float16)

strided_search_range = search_range * dilation_rate
padded_lvl = tf.pad(c1, [[0, 0],
                         [strided_search_range, strided_search_range],
                         [strided_search_range, strided_search_range],
                         [0, 0]])
_, h, w, c = c1.get_shape().as_list()
c1_nchw = c1  # tf.transpose(c1, perm=[0, 3, 1, 2])
pl_nchw = padded_lvl  # tf.transpose(padded_lvl, perm=[0, 3, 1, 2])
_, h_pad, w_pad, _ = pl_nchw.shape


def patches_indices_meshgrid(image, patch_h, patch_w):
    b, h, w, c = image.shape
    indices = [list(tf.convert_to_tensor(
        tf.meshgrid(tf.range(b), tf.range(i, i + patch_h),
                    tf.range(j, j + patch_w), tf.range(c)))) for i in
               range(0, h - patch_h + 1) for j in range(0, w - patch_w + 1)]
    indices = tf.convert_to_tensor(indices)
    # The indices are correct but we need to transpose to have the same order as extract_patches
    indices = tf.transpose(indices, [0, 3, 2, 4, 5, 1])
    indices = tf.reshape(indices, [indices.shape[0], -1, indices.shape[-1]])
    return indices

def patches_indices_meshgrid_no_b(image, patch_h, patch_w):
    _, h, w, c = image.shape
    indices = [list(tf.convert_to_tensor(
        tf.meshgrid(tf.range(i, i + patch_h),
                    tf.range(j, j + patch_w),
                    tf.range(c)))) for i in range(0, h - patch_h + 1) for j in range(0, w - patch_w + 1)]
    indices = tf.convert_to_tensor(indices)

    # The indices are correct but we need to transpose to have the same order as extract_patches
    indices = tf.transpose(indices, [0, 3, 2, 4, 1])
    indices = tf.reshape(indices, [indices.shape[0], -1, indices.shape[-1]])
    return indices

indices = patches_indices_meshgrid(pl_nchw, h, w)
patches = tf.gather_nd(pl_nchw, indices, batch_dims=0)

indices2 = patches_indices_meshgrid_no_b(pl_nchw, h, w)
patches2=tf.map_fn(lambda x: tf.gather_nd(x, indices2, batch_dims=0), pl_nchw, dtype=tf.float16)
print(patches2.dtype)
patches2 = tf.transpose(patches2, perm=[1, 0,2])
patches2 = tf.reshape(patches2, [patches2.shape[0], -1])
print(patches2.shape)
print(tf.reduce_all(tf.math.equal(patches, patches2)))

patches = tf.reshape(patches, [search_range*2+1, search_range*2+1, -1, h, w, c])
patches = tf.transpose(patches, perm=[2, 0, 1, 3, 4, 5])
print(patches2.shape)
patches2 = tf.reshape(patches2, [search_range*2+1, search_range*2+1, -1, h, w, c])
patches2 = tf.transpose(patches2, perm=[2, 0, 1, 3, 4, 5])
print(patches2.shape)
print(tf.reduce_all(tf.math.equal(patches, patches2)))

# c1_repeat = tf.expand_dims(c1,axis=1)
# c1_repeat = tf.expand_dims(c1_repeat,axis=1)
# c1_repeat = tf.tile(c1_repeat, [1,patches.shape[1], patches.shape[2],1, 1, 1])
#
# patches = c1_repeat*patches
# split_stack = tf.stack(tf.split(patches, num_or_size_splits=nbre_cuts, axis=-1), axis=-1)
# cost = tf.reduce_mean(split_stack, axis=-2)
# cost = tf.transpose(cost, perm=[0, 3, 4, 1, 2, 5])
# cost_vol2 = tf.reshape(cost, cost.shape[0:1] + cost.shape[1:3] + [cost.shape[3]*cost.shape[4]*cost.shape[5]])
