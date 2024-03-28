import onnxruntime
import onnx
# Path to the ONNX model file
onnx_model_path = 'model/resnet-34_kinetics .onnx'
hh=onnx.load(onnx_model_path)
# Create an ONNX runtime session
session = onnxruntime.InferenceSession(onnx_model_path)

# Get the input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Print input and output details
print(hh)
print("Input name:", input_name)
print("Input shape:", session.get_inputs()[0].shape)
print("Output name:", output_name)
print("Output shape:", session.get_outputs()[0].shape)
