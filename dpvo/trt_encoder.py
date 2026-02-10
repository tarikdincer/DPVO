#!/usr/bin/env python3
"""
TensorRT Inference Wrapper for DPVO Encoders
=============================================

Drop-in replacement for BasicEncoder4 using TensorRT engines.
Use this in your DPVO pipeline on Jetson AGX Orin.

Usage:
    from trt_encoder import TRTEncoder

    fnet = TRTEncoder("fnet_encoder.trt")
    inet = TRTEncoder("inet_encoder.trt")

    # Same interface as original — pass (b, n, 3, H, W) tensor
    fmap = fnet(images) / 4.0
    imap = inet(images) / 4.0
"""

import torch
import numpy as np

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("WARNING: tensorrt not available. Install on Jetson with: sudo apt install python3-libnvinfer")


class TRTEncoder:
    """TensorRT drop-in replacement for BasicEncoder4."""

    def __init__(self, engine_path, device="cuda:0"):
        assert TRT_AVAILABLE, "tensorrt package required"

        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Get binding info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)

        # Pre-allocate output buffer
        self.output_buffer = torch.empty(
            *self.output_shape, dtype=torch.float32, device=self.device
        )

        print(f"Loaded TRT engine: {engine_path}")
        print(f"  Input:  {self.input_name} {list(self.input_shape)}")
        print(f"  Output: {self.output_name} {list(self.output_shape)}")

    def infer(self, input_tensor):
        """Run inference on a (1, 3, H, W) tensor. Returns (1, C, H/4, W/4)."""
        assert input_tensor.is_cuda and input_tensor.is_contiguous()

        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_buffer.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)
        torch.cuda.current_stream().synchronize()

        return self.output_buffer.clone()

    def __call__(self, images):
        """
        Drop-in replacement for BasicEncoder4.forward().
        Accepts (b, n, 3, H, W) — same as the original.
        Returns (b, n, C, H/4, W/4).
        """
        b, n, c, h, w = images.shape
        imgs_flat = images.reshape(b * n, c, h, w).contiguous()

        # Process each image (batch=1 engine)
        # For multi-batch, rebuild engine with batch>1 or use dynamic shapes
        outputs = []
        for i in range(b * n):
            out = self.infer(imgs_flat[i : i + 1])
            outputs.append(out)

        output = torch.cat(outputs, dim=0)
        _, c_out, h_out, w_out = output.shape
        return output.view(b, n, c_out, h_out, w_out)


class TRTPatchifier:
    """
    Partial replacement for DPVO's Patchifier.
    Replaces fnet/inet with TensorRT, keeps patchification in PyTorch.
    
    Usage in dpvo.py:
        # Replace:
        #   self.network.patchify(images, ...)
        # With:
        #   self.trt_patchifier(images, self.network.patchify, ...)
    """

    def __init__(self, fnet_engine_path, inet_engine_path):
        self.fnet = TRTEncoder(fnet_engine_path)
        self.inet = TRTEncoder(inet_engine_path)

    def __call__(self, patchifier, images, **kwargs):
        """
        Run TRT encoders, then use original patchifier for patch extraction.
        
        Args:
            patchifier: The original Patchifier module (for altcorr.patchify calls)
            images: (b, n, 3, H, W) normalized images (already 2*(img/255)-0.5)
        """
        # Use TRT for the heavy CNN encoding
        fmap = self.fnet(images) / 4.0
        imap = self.inet(images) / 4.0

        # The rest (patchify, coordinate sampling) stays in PyTorch
        # This mirrors Patchifier.forward() after the fnet/inet calls
        b, n, c, h, w = fmap.shape
        P = patchifier.patch_size

        patches_per_image = kwargs.get('patches_per_image', 80)
        disps = kwargs.get('disps', None)
        return_color = kwargs.get('return_color', False)

        from dpvo import altcorr
        from dpvo.utils import coords_grid_with_index

        x = torch.randint(1, w - 1, size=[n, patches_per_image], device="cuda")
        y = torch.randint(1, h - 1, size=[n, patches_per_image], device="cuda")
        coords = torch.stack([x, y], dim=-1).float()

        imap_patches = altcorr.patchify(imap[0], coords, 0).view(b, -1, 384, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, P // 2).view(b, -1, 128, P, P)

        if return_color:
            clr = altcorr.patchify(images[0], 4 * (coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, P // 2).view(b, -1, 3, P, P)

        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap, imap_patches, patches, index, clr

        return fmap, gmap, imap_patches, patches, index