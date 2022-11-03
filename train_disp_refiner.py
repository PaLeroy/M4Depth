from tensorflow import keras as ks, float32
import tensorflow as tf
from m4depth_network_funct import disp_refiner_as_a_function
from m4depth_network import DispRefiner
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
if __name__ == '__main__':
    x1 = ks.Input((48, 48, 122), dtype=float32)
    x = disp_refiner_as_a_function(0, "disp", x1)
    model = ks.Model(inputs=[x1], outputs=[x])
    print(model.inputs)
    print(model.outputs)

    x2 = DispRefiner(regularizer_weight=0)(x1)
    model2 = ks.Model(inputs=[x1], outputs=[x2])
    print(model2.inputs)
    print(model2.outputs)

    x_train = tf.random.uniform((1000, 48, 48, 122), dtype=float32)

    y_train_1 = tf.ones((1000, 48, 48, 5), dtype=float32)
    y_train_2 = tf.zeros((1000, 48, 48, 96), dtype=float32)

    # print(model2(x_train[0:3])[0])

    model.compile(
        optimizer=ks.optimizers.RMSprop(),
        loss=[ks.losses.MeanSquaredError(), ks.losses.MeanSquaredError()]
    )

    model.fit([x_train],
              [[y_train_1, y_train_2]],
              batch_size=32,
              epochs=1000)