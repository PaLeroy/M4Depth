import tf2onnx
import onnx
from tensorflow import keras

from m4depth_network_funct import DomainNormalization, log_tensor_companion, \
    get_disparity_sweeping_cv, cost_volume,prev_d2disp,disp2depth

custom_objects = {"DomainNormalization": DomainNormalization,
                  "log_tensor_companion": log_tensor_companion,
                  "get_disparity_sweeping_cv": get_disparity_sweeping_cv,
                  "cost_volume": cost_volume,
                  "prev_d2disp": prev_d2disp,
                  "disp2depth": disp2depth}

model_path = 'm4depth_model_L_2.h5'
with keras.utils.custom_object_scope(custom_objects):
    keras_model = keras.models.load_model(model_path)

model_proto, _ = tf2onnx.convert.from_keras(keras_model, opset=13, output_path="test_onnx")
print(type(model_proto))
onnx.checker.check_model(model_proto)