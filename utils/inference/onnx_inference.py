import tensorrt as trt


model_path = '/home/amicro/project/yolov3/runs/train/exp5/weights/best.onnx'
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
	with open(model_path, 'rb') as model:
		if not parser.parse(model.read()):
			for error in range(parser.num_errors):
				print(parser.get_error(error))