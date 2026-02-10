#!/usr/bin/env python3
"""
TensorRT Update Network Integration for DPVO (v2 — fixed)
==========================================================

CRITICAL FIX from v1: c1/c2 neighbor MLPs now run in PyTorch (Phase B),
not TRT, because they must index into the UPDATED net (after corr+norm).

Architecture:
    Phase A (TRT):     net + inp + corr_enc(corr), then norm
    Phase B (PyTorch): c1(net[:,ix]), c2(net[:,jx]), SoftAgg
    Phase C (TRT):     GRU + delta/weight heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

DIM = 384


# ─────────────────────────────────────────────────────────
# Pure PyTorch SoftAgg
# ─────────────────────────────────────────────────────────

class SoftAggPure(nn.Module):
    def __init__(self, dim=384, expand=True):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)

    def forward(self, x, ix):
        x = x.float()
        E = x.shape[1]

        _, jx = torch.unique(ix, return_inverse=True)
        num_groups = jx.max().item() + 1
        jx_3d = jx.view(1, E, 1).expand(1, E, self.dim)

        gx = self.g(x).float()
        max_vals = torch.full((1, num_groups, self.dim), -1e9, device=x.device, dtype=torch.float32)
        max_vals.scatter_reduce_(1, jx_3d, gx, reduce='amax', include_self=False)

        gx_shifted = gx - max_vals.gather(1, jx_3d)
        exp_gx = torch.exp(gx_shifted)

        sum_exp = torch.zeros(1, num_groups, self.dim, device=x.device, dtype=torch.float32)
        sum_exp.scatter_add_(1, jx_3d, exp_gx)

        w = exp_gx / sum_exp.gather(1, jx_3d).clamp(min=1e-8)

        fx = self.f(x).float()
        y = torch.zeros(1, num_groups, self.dim, device=x.device, dtype=torch.float32)
        y.scatter_add_(1, jx_3d, fx * w)

        if self.expand:
            return self.h(y).gather(1, jx_3d)
        return self.h(y)


# ─────────────────────────────────────────────────────────
# PyTorch c1/c2 modules (loaded from checkpoint)
# ─────────────────────────────────────────────────────────

class NeighborMLP(nn.Module):
    """Standalone c1 or c2 module."""
    def __init__(self, dim=384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim))

    def forward(self, x):
        return self.mlp(x)


# ─────────────────────────────────────────────────────────
# TRT Engine Wrapper (dynamic shapes with pad/chunk)
# ─────────────────────────────────────────────────────────

class TRTDynamic:
    def __init__(self, engine_path):
        assert TRT_AVAILABLE, "tensorrt required"
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.num_io = self.engine.num_io_tensors
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.num_io)]
        self.input_names = [n for n in self.tensor_names
                           if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.tensor_names
                            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        self.min_edge_dim = None
        self.max_edge_dim = None
        num_profiles = self.engine.num_optimization_profiles
        if num_profiles > 0 and len(self.input_names) > 0:
            first_input = self.input_names[0]
            min_shape, _, max_shape = self.engine.get_tensor_profile_shape(first_input, 0)
            if len(min_shape) >= 2:
                self.min_edge_dim = min_shape[1]
            if len(max_shape) >= 2:
                self.max_edge_dim = max_shape[1]

        print(f"Loaded TRT engine: {engine_path}")
        print(f"  Inputs:  {self.input_names}")
        print(f"  Outputs: {self.output_names}")
        print(f"  Edge dim range: [{self.min_edge_dim}, {self.max_edge_dim}]")

    def _infer(self, *inputs):
        stream = torch.cuda.current_stream().cuda_stream
        for name, tensor in zip(self.input_names, inputs):
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())

        outputs = []
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            out = torch.empty(*shape, dtype=torch.float32, device="cuda")
            self.context.set_tensor_address(name, out.data_ptr())
            outputs.append(out)

        self.context.execute_async_v3(stream)
        torch.cuda.current_stream().synchronize()
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def _pad_inputs(self, inputs, target_E):
        padded = []
        for t in inputs:
            E = t.shape[1]
            if E >= target_E:
                padded.append(t)
            else:
                pad = torch.zeros(t.shape[0], target_E - E, *t.shape[2:],
                                  dtype=t.dtype, device=t.device)
                padded.append(torch.cat([t, pad], dim=1).contiguous())
        return padded

    def _slice_outputs(self, result, E):
        if isinstance(result, tuple):
            return tuple(r[:, :E].contiguous() for r in result)
        return result[:, :E].contiguous()

    def __call__(self, *inputs):
        E = inputs[0].shape[1]
        min_E = self.min_edge_dim or 1
        max_E = self.max_edge_dim or 999999

        if min_E <= E <= max_E:
            return self._infer(*inputs)

        if E < min_E:
            padded = self._pad_inputs(inputs, min_E)
            result = self._infer(*padded)
            return self._slice_outputs(result, E)

        # Chunking for E > max_E
        all_chunk_outputs = None
        for s in range(0, E, max_E):
            e = min(s + max_E, E)
            chunk_inputs = [t[:, s:e].contiguous() for t in inputs]
            chunk_E = e - s

            if chunk_E < min_E:
                chunk_inputs = self._pad_inputs(chunk_inputs, min_E)
                chunk_result = self._infer(*chunk_inputs)
                chunk_result = self._slice_outputs(chunk_result, chunk_E)
            else:
                chunk_result = self._infer(*chunk_inputs)

            if all_chunk_outputs is None:
                if isinstance(chunk_result, tuple):
                    all_chunk_outputs = [[r] for r in chunk_result]
                else:
                    all_chunk_outputs = [[chunk_result]]
            else:
                if isinstance(chunk_result, tuple):
                    for i, r in enumerate(chunk_result):
                        all_chunk_outputs[i].append(r)
                else:
                    all_chunk_outputs[0].append(chunk_result)

        concatenated = [torch.cat(chunks, dim=1) for chunks in all_chunk_outputs]
        return concatenated[0] if len(concatenated) == 1 else tuple(concatenated)


# ─────────────────────────────────────────────────────────
# Hybrid TRT Update
# ─────────────────────────────────────────────────────────

class TRTUpdate:
    """
    Drop-in replacement for self.network.update().
    
    Faithfully reproduces the original forward():
        net = net + inp + self.corr(corr)          # Phase A (TRT)
        net = self.norm(net)                         # Phase A (TRT)
        ix, jx = fastba.neighbors(kk, jj)           # Phase B (PyTorch)
        net = net + self.c1(mask_ix * net[:,ix])     # Phase B (PyTorch)
        net = net + self.c2(mask_jx * net[:,jx])     # Phase B (PyTorch)
        net = net + self.agg_kk(net, kk)             # Phase B (PyTorch)
        net = net + self.agg_ij(net, ii*12345+jj)    # Phase B (PyTorch)
        net = self.gru(net)                          # Phase C (TRT)
        return net, (self.d(net), self.w(net), None) # Phase C (TRT)
    """

    def __init__(self, phase_a_engine, phase_c_engine, state_dict=None, update_prefix="update."):
        # TRT engines
        self.phase_a = TRTDynamic(phase_a_engine)
        self.phase_c = TRTDynamic(phase_c_engine)

        # PyTorch modules for Phase B (c1, c2, SoftAgg)
        self.c1 = NeighborMLP(DIM).float().cuda().eval()
        self.c2 = NeighborMLP(DIM).float().cuda().eval()
        self.agg_kk = SoftAggPure(DIM, expand=True).float().cuda().eval()
        self.agg_ij = SoftAggPure(DIM, expand=True).float().cuda().eval()

        if state_dict is not None:
            self._load_phase_b_weights(state_dict, update_prefix)

        try:
            from dpvo import fastba
            self.fastba = fastba
        except ImportError:
            self.fastba = None

    def _load_phase_b_weights(self, state_dict, prefix):
        """Load c1, c2, agg_kk, agg_ij weights from checkpoint."""
        
        # c1: Sequential(Linear, ReLU, Linear) → NeighborMLP.mlp = Sequential(Linear, ReLU, Linear)
        # Keys: c1.0.weight/bias, c1.2.weight/bias → mlp.0.weight/bias, mlp.2.weight/bias
        for c_name, c_module in [("c1", self.c1), ("c2", self.c2)]:
            c_sd = {}
            for orig_idx, new_idx in [(0, 0), (2, 2)]:
                for suffix in ["weight", "bias"]:
                    key = f"{prefix}{c_name}.{orig_idx}.{suffix}"
                    if key in state_dict:
                        c_sd[f"mlp.{new_idx}.{suffix}"] = state_dict[key]
            if c_sd:
                c_module.load_state_dict(c_sd, strict=False)
                print(f"  Loaded {len(c_sd)} params for {c_name}")
            else:
                print(f"  Warning: no weights found for {c_name}")

        # SoftAgg
        for agg_name, agg_module in [("agg_kk", self.agg_kk), ("agg_ij", self.agg_ij)]:
            agg_sd = {}
            for param in ["f.weight", "f.bias", "g.weight", "g.bias", "h.weight", "h.bias"]:
                key = f"{prefix}{agg_name}.{param}"
                if key in state_dict:
                    agg_sd[param] = state_dict[key]
            if agg_sd:
                agg_module.load_state_dict(agg_sd, strict=False)
                print(f"  Loaded {len(agg_sd)} params for {agg_name}")
            else:
                print(f"  Warning: no weights found for {agg_name}")

    def __call__(self, net, inp, corr, flow, ii, jj, kk):
        """Exact same signature as Update.forward()."""

        # ── Phase A (TRT): corr encoding + norm ──
        net = self.phase_a(
            net.contiguous().float(),
            inp.contiguous().float(),
            corr.contiguous().float(),
        )

        # ── Phase B (PyTorch): c1, c2 on UPDATED net + SoftAgg ──
        ix, jx = self.fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        ix_safe = ix.clamp(min=0)
        jx_safe = jx.clamp(min=0)

        # c1 and c2 now operate on the UPDATED net (after Phase A)
        net = net + self.c1(mask_ix * net[:, ix_safe])
        net = net + self.c2(mask_jx * net[:, jx_safe])

        with torch.no_grad():
            net = net + self.agg_kk(net, kk)
            net = net + self.agg_ij(net, ii * 12345 + jj)

        # ── Phase C (TRT): GRU + output heads ──
        result_c = self.phase_c(net.contiguous().float())

        if isinstance(result_c, tuple):
            net_out, delta, weight = result_c
        else:
            net_out = result_c
            delta = net_out[..., :2]
            weight = torch.sigmoid(net_out[..., :2])

        return net_out, (delta, weight, None)


# ─────────────────────────────────────────────────────────
# Pure PyTorch fallback (for testing)
# ─────────────────────────────────────────────────────────

class PurePytorchUpdate:
    def __init__(self, checkpoint_path, update_prefix="update."):
        import sys
        sys.path.insert(0, '.')
        from dpvo.net import Update
        self.update_module = Update(p=3).cuda().eval()
        
        sd = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]

        update_sd = {k[len(update_prefix):]: v for k, v in sd.items() if k.startswith(update_prefix)}
        self.update_module.load_state_dict(update_sd, strict=False)
        print(f"Loaded {len(update_sd)} params into Update module")

    def __call__(self, net, inp, corr, flow, ii, jj, kk):
        with torch.no_grad():
            return self.update_module(net, inp, corr, flow, ii, jj, kk)