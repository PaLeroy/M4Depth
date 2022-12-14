import tf2onnx
import onnx
from onnxruntime import InferenceSession
from tensorflow import keras
import onnxmltools
import numpy as np
from m4depth_network_funct import RescaleLayer, log_tensor_companion, \
    get_disparity_sweeping_cv, cost_volume,prev_d2disp,disp2depth

custom_objects = {"RescaleLayer": RescaleLayer,}
model_path = 'm4depth_model_L_2_old_v_False.h5'
with keras.utils.custom_object_scope(custom_objects):
    keras_model = keras.models.load_model(model_path)

model_proto, _ = tf2onnx.convert.from_keras(keras_model, opset=13, output_path="test_onnx")
onnx.checker.check_model(model_proto)
onnxmltools.save_model(model_proto, "model_proto.onnx")

onnx_model=model_proto
def shape2tuple(shape):
    return tuple(getattr(d, 'dim_value', 0) for d in shape.dim)

print('** inputs **')
for obj in onnx_model.graph.input:
    print("name=%r dtype=%r shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))


b=4
image=np.random.rand(b, 384, 384, 3).astype(np.float32)
camera_f_input=np.random.rand(b, 2).astype(np.float32)
camera_c_input=np.random.rand(b, 2).astype(np.float32)
rot_input=np.random.rand(b, 4).astype(np.float32)
trans_input=np.random.rand(b, 3).astype(np.float32)
L_2_disp_L1_t=np.random.rand(b, 96, 96, 1).astype(np.float32)
L_2_other_L1_t=np.random.rand(b, 96, 96, 4).astype(np.float32)
L_2_f_enc_L_t1=np.random.rand(b, 96, 96, 32).astype(np.float32)
L_2_d_est_L_t1=np.random.rand(b, 96, 96, 1).astype(np.float32)
L_1_f_enc_L_t1=np.random.rand(b, 192, 192, 16).astype(np.float32)
L_1_d_est_L_t1=np.random.rand(b, 192, 192, 1).astype(np.float32)
inputs_dict={
    'image': image,
    'camera_f_input': camera_f_input,
    'camera_c_input': camera_c_input,
    'rot_input': rot_input,
    'trans_input': trans_input,
    'L_2_disp_L-1_t': L_2_disp_L1_t,
    'L_2_other_L-1_t': L_2_other_L1_t,
    'L_2_f_enc_L_t-1': L_2_f_enc_L_t1,
    'L_2_d_est_L_t-1': L_2_d_est_L_t1,
    'L_1_f_enc_L_t-1': L_1_f_enc_L_t1,
    'L_1_d_est_L_t-1': L_1_d_est_L_t1
}
sess = InferenceSession(model_proto.SerializeToString())
res = sess.run(None, inputs_dict)
print('** outputs **')
for obj, r in zip(onnx_model.graph.output, res):
    print("name=%r dtype=%r shape=%r" % (
        obj.name, obj.type.tensor_type.elem_type,
        shape2tuple(obj.type.tensor_type.shape)))
    print(r.shape)

#TODO: faire l'eval loop!

# # the list of outputs
# print('** outputs **')
# print(onnx_model.graph.output)
#
# # in a more nicely format
# print('** outputs **')
# for obj in onnx_model.graph.output:
#     print("name=%r dtype=%r shape=%r" % (
#         obj.name, obj.type.tensor_type.elem_type,
#         shape2tuple(obj.type.tensor_type.shape)))
#
# # the list of nodes
# print('** nodes **')
# print(onnx_model.graph.node)
#
# # in a more nicely format
print('** nodes **')
for node in onnx_model.graph.node:
    print("name=%r type=%r input=%r output=%r" % (
        node.name, node.op_type, node.input, node.output))