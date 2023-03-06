import tensorflow as tf
c1=tf.random.uniform((32, 21, 21, 13))
max_offset=7
dilation_rate=1
search_range=2
nbre_cuts=2

c1 = tf.cast(c1, tf.float16)

strided_search_range = search_range * dilation_rate
padded_lvl = tf.pad(c1, [[0, 0],
                         [strided_search_range, strided_search_range],
                         [strided_search_range, strided_search_range],
                         [0, 0]])
_, h, w, c = c1.get_shape().as_list()
c1_nchw = c1 #tf.transpose(c1, perm=[0, 3, 1, 2])
pl_nchw = padded_lvl #tf.transpose(padded_lvl, perm=[0, 3, 1, 2])
_, h_pad, w_pad, _ = pl_nchw.shape
list_all2=tf.image.extract_patches(pl_nchw,
                                   sizes=[1, h, w, 1],
                                   strides=[1, dilation_rate, dilation_rate,1],
                                   rates=[1,1,1,1],
                                   padding="VALID")

list_all2=tf.reshape(list_all2, list_all2.shape[0:3]+c1.shape[1:4])

def patches_indices_meshgrid(image, patch_h, patch_w):
    b, h, w, c = image.shape
    indices=[list(tf.convert_to_tensor(tf.meshgrid(tf.range(b), tf.range(i, i + patch_h), tf.range(j, j + patch_w), tf.range(c)))) for i in range(0, h-patch_h+1) for j in range(0, w-patch_w+1)]
    indices=tf.convert_to_tensor(indices)
    # The indices are correct but we need to transpose to have the same order as extract_patches
    indices = tf.transpose(indices, [0, 3, 2, 4, 5, 1])
    indices = tf.reshape(indices, [indices.shape[0], -1, indices.shape[-1]])
    return indices
p_h=h
p_w=w

indices = patches_indices_meshgrid(pl_nchw, p_h, p_w)
patches = tf.gather_nd(pl_nchw, indices, batch_dims=0)
patches = tf.reshape(patches, [search_range*2+1, search_range*2+1, -1, h, w, c])
patches = tf.transpose(patches, perm=[2, 0, 1, 3, 4, 5])
print("list", patches.shape)
print("list2=", list_all2.shape)

print(tf.reduce_all(tf.math.equal(list_all2, patches)))


# c1_repeat = tf.expand_dims(c1,axis=1)
# c1_repeat = tf.expand_dims(c1_repeat,axis=1)
# c1_repeat = tf.tile(c1_repeat, [1,list_all2.shape[1], list_all2.shape[2],1, 1, 1])
# list_all2 = c1_repeat*list_all2
# split_stack = tf.stack(tf.split(list_all2, num_or_size_splits=nbre_cuts, axis=-1), axis=-1)
# cost = tf.reduce_mean(split_stack, axis=-2)
# cost = tf.transpose(cost, perm=[0, 3, 4, 1, 2, 5])
# cost_vol = tf.reshape(cost, cost.shape[0:1] + cost.shape[1:3] + [cost.shape[3]*cost.shape[4]*cost.shape[5]])
# print("final correct cost_vol", cost_vol.shape)