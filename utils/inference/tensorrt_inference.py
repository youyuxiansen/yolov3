import tensorrt as trt
import numpy as np
import cv2

import pycuda.driver as cuda
import pycuda.autoinit


class HostDeviceMem(object):
	def __init__(self, host_mem, device_mem):
		self.host = host_mem
		self.device = device_mem

	def __str__(self):
		return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

	def __repr__(self):
		return self.__str__()


class TrtModel:

	def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):

		self.engine_path = engine_path
		self.dtype = dtype
		self.logger = trt.Logger(trt.Logger.WARNING)
		self.runtime = trt.Runtime(self.logger)
		self.engine = self.load_engine(self.runtime, self.engine_path)
		self.max_batch_size = max_batch_size
		self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
		self.context = self.engine.create_execution_context()

	@staticmethod
	def load_engine(trt_runtime, engine_path):
		trt.init_libnvinfer_plugins(None, "")
		with open(engine_path, 'rb') as f:
			engine_data = f.read()
		engine = trt_runtime.deserialize_cuda_engine(engine_data)
		return engine

	def allocate_buffers(self):

		inputs = []
		outputs = []
		bindings = []
		stream = cuda.Stream()

		for binding in self.engine:
			size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
			host_mem = cuda.pagelocked_empty(size, self.dtype)
			device_mem = cuda.mem_alloc(host_mem.nbytes)

			bindings.append(int(device_mem))

			if self.engine.binding_is_input(binding):
				inputs.append(HostDeviceMem(host_mem, device_mem))
			else:
				outputs.append(HostDeviceMem(host_mem, device_mem))

		return inputs, outputs, bindings, stream

	def __call__(self, x: np.ndarray, batch_size=2):

		x = x.astype(self.dtype)

		np.copyto(self.inputs[0].host, x.ravel())

		for inp in self.inputs:
			cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

		self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
		for out in self.outputs:
			cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

		self.stream.synchronize()
		return [out.host.reshape(batch_size, -1) for out in self.outputs]


if __name__ == "__main__":
	batch_size = 1
	trt_engine_path = '/home/amicro/project/yolov3/runs/train/exp5/weights/best.trt'
	model = TrtModel(trt_engine_path)
	shape = model.engine.get_binding_shape(0)
	data = cv2.imread('/home/amicro/下载/5474.jpg')
	h0, w0 = data.shape[:2]
	r = shape[2] / max(h0, w0)  # resize image to img_size
	if r != 1:  # always resize down, only resize up if training with augmentation
		interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
		data = cv2.resize(data, (int(shape[2]), int(shape[3])), interpolation=interp)
	data = np.expand_dims(data, 0)
	data = data.transpose(0, 3, 1, 2) / 225
	data = np.random.randint(0, 255, (batch_size, *shape[1:])) / 255
	result = model(data, batch_size)
	print(result)
	cv2.imwrite('/home/amicro/下载/5474_a.jpg', result[0])
