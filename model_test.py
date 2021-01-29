from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter('model/ssd_mobilenet_v2_320x320_coco17_tpu-8.tflite')
interpreter.allocate_tensors()

print(interpreter.get_input_details())
print(len(interpreter.get_output_details()))