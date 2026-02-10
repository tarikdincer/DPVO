#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import tensorrt as trt
import weakref

from cuda.bindings import runtime as cudart


def _cleanup_cuda_resources(d_input, d_output, stream):
    """cleanup function for CUDA resources"""
    if d_input is not None:
        cudart.cudaFree(d_input)
    if d_output is not None:
        cudart.cudaFree(d_output)
    if stream is not None:
        cudart.cudaStreamDestroy(stream)


class ONNXClassifierWrapper:
    def __init__(self, file, target_dtype=np.float32):
        self.stream = None
        self.d_input = None
        self.d_output = None

        self.target_dtype = target_dtype
        self.num_classes = 1000
        self.load(file)

        self._finalizer = weakref.finalize(self, _cleanup_cuda_resources, self.d_input, self.d_output, self.stream)

    def load(self, file):
        with open(file, "rb") as f:
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

    def allocate_memory(self, batch):
        self.output = np.empty(
            self.num_classes, dtype=self.target_dtype
        )  # Need to set both input and output precisions to FP16 to fully enable FP16

        # allocate device memory
        err, self.d_input = cudart.cudaMalloc(batch.nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to allocate input memory: {cudart.cudaGetErrorString(err)}")

        err, self.d_output = cudart.cudaMalloc(self.output.nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to allocate output memory: {cudart.cudaGetErrorString(err)}")

        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        assert len(tensor_names) == 2

        self.context.set_tensor_address(tensor_names[0], int(self.d_input))
        self.context.set_tensor_address(tensor_names[1], int(self.d_output))

        err, self.stream = cudart.cudaStreamCreate()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to create stream: {cudart.cudaGetErrorString(err)}")

        # update the finalizer with new resources
        self._finalizer.detach()
        self._finalizer = weakref.finalize(self, _cleanup_cuda_resources, self.d_input, self.d_output, self.stream)

    def predict(self, batch):  # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)

        # transfer input data to device
        err = cudart.cudaMemcpyAsync(
            self.d_input, batch.ctypes.data, batch.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to copy input to device: {cudart.cudaGetErrorString(err)}")

        # execute model
        self.context.execute_async_v3(self.stream)

        # transfer predictions back
        err = cudart.cudaMemcpyAsync(
            self.output.ctypes.data,
            self.d_output,
            self.output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream,
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to copy output from device: {cudart.cudaGetErrorString(err)}")

        # synchronize threads
        err = cudart.cudaStreamSynchronize(self.stream)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to synchronize stream: {cudart.cudaGetErrorString(err)}")

        return self.output

    def cleanup(self):
        """free allocated CUDA memory and destroy stream"""
        if hasattr(self, "_finalizer"):
            self._finalizer()

        self.d_input = None
        self.d_output = None
        self.stream = None


def convert_onnx_to_engine(onnx_filename, engine_filename=None, max_workspace_size=1 << 30, fp16_mode=True):
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(logger) as builder, builder.create_network() as network, trt.OnnxParser(
        network, logger
    ) as parser, builder.create_builder_config() as builder_config:
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        if fp16_mode:
            builder_config.set_flag(trt.BuilderFlag.FP16)

        print("Parsing ONNX file.")
        with open(onnx_filename, "rb") as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        print("Building TensorRT engine. This may take a few minutes.")
        serialized_engine = builder.build_serialized_network(network, builder_config)

        if engine_filename:
            with open(engine_filename, "wb") as f:
                f.write(serialized_engine)

        return serialized_engine, logger

def _cleanup_multi(d_ptrs, stream):
    for p in d_ptrs.values():
        if p is not None:
            cudart.cudaFree(p)
    if stream is not None:
        cudart.cudaStreamDestroy(stream)

class TRTEngineMultiIO:
    def __init__(self, engine_file):
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_file, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [n for n in self.tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        self.dtypes = {n: trt.nptype(self.engine.get_tensor_dtype(n)) for n in self.tensor_names}
        self.d_ptrs = {}
        self.shapes = {}
        self.h_out = {}

        err, self.stream = cudart.cudaStreamCreate()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("cudaStreamCreate failed")

        self._finalizer = weakref.finalize(self, _cleanup_multi, self.d_ptrs, self.stream)

    def _ensure(self, name, shape):
        shape = tuple(int(x) for x in shape)
        dtype = self.dtypes[name]
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

        if self.shapes.get(name) == shape and name in self.d_ptrs:
            return

        if name in self.d_ptrs:
            cudart.cudaFree(self.d_ptrs[name])

        err, dptr = cudart.cudaMalloc(nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc failed for {name}")
        self.d_ptrs[name] = dptr
        self.shapes[name] = shape

        # bind pointer
        self.context.set_tensor_address(name, int(dptr))

        if name in self.output_names:
            self.h_out[name] = np.empty(shape, dtype=dtype)

    def infer(self, inputs: dict):
        # set shapes for inputs (dynamic)
        for name in self.input_names:
            arr = inputs[name]
            if arr.dtype != self.dtypes[name]:
                arr = arr.astype(self.dtypes[name], copy=False)
                inputs[name] = arr
            self.context.set_input_shape(name, arr.shape)

        # alloc all tensors
        for name in self.tensor_names:
            shape = self.context.get_tensor_shape(name)
            self._ensure(name, shape)

        # H2D
        for name in self.input_names:
            arr = inputs[name]
            err = cudart.cudaMemcpyAsync(
                self.d_ptrs[name],
                arr.ctypes.data,
                arr.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"H2D failed for {name}")

        # run
        self.context.execute_async_v3(self.stream)

        # D2H
        outputs = {}
        for name in self.output_names:
            out = self.h_out[name]
            err = cudart.cudaMemcpyAsync(
                out.ctypes.data,
                self.d_ptrs[name],
                out.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"D2H failed for {name}")
            outputs[name] = out

        cudart.cudaStreamSynchronize(self.stream)
        return outputs
