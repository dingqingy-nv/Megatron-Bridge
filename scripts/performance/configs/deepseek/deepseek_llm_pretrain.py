# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import torch
from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.deepseek.deepseek_v3 import (
    deepseek_v3_pretrain_config as pretrain_config,
)
from megatron.bridge.recipes.deepseek.deepseek_v3 import (
    set_deepseek_v3_pipeline_model_parallel_layout,
)
from megatron.bridge.recipes.deepseek.deepseek_v4 import (
    deepseek_v4_pro_pretrain_mxfp8_config,
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.cuda_graph import is_full_iteration_cuda_graph


logger = logging.getLogger(__name__)


def set_deepseek_v3_common_configs(cfg: ConfigContainer, moe_a2a_overlap: bool = False) -> None:
    """Set common performance configurations for all DeepSeek-V3 configs."""
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.model.moe_router_fusion = True
    cfg.model.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True


def set_full_iter_cg_configs(cfg: ConfigContainer) -> None:
    """Apply defaults required by full-iteration CUDA graph capture with dropless MoE.

    Dropless MoE produces variable-shaped per-expert tensors that CG cannot
    capture; we pad to a fixed capacity (pad_experts + capacity factor) and use
    MCore PR #4247 paged stashing to recover memory. Callers should gate on
    `is_full_iteration_cuda_graph(cfg.model)`.
    """
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0


def deepseek_v3_pretrain_config_gb300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="gb300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    if cfg.mixed_precision.fp8_recipe == "mxfp8":
        cfg.model.fp8_output_proj = True

    # Apply model-specific settings that were previously passed as constructor args
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute layout based on updated PP/VP sizes
        set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_pretrain_config_gb200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    if cfg.mixed_precision.fp8_recipe == "mxfp8":
        cfg.model.fp8_output_proj = True

    # Apply model-specific settings that were previously passed as constructor args
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute layout based on updated PP/VP sizes
        set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_pretrain_config_vr200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v2"
) -> ConfigContainer:
    """VR200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="vr200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    # Apply model-specific settings that were previously passed as constructor args
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute layout based on updated PP/VP sizes
        set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_pretrain_config_b300(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B300, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="b300",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    # Apply model-specific settings that were previously passed as constructor args
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    # Recompute layout based on updated PP/VP sizes
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v3_pretrain_config_b200(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """B200, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="b200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    # Apply model-specific settings that were previously passed as constructor args
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    # Recompute layout based on updated PP/VP sizes
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    cfg.comm_overlap.overlap_grad_reduce = True

    cfg.mixed_precision.fp4_param_gather = False

    return cfg


def deepseek_v3_pretrain_config_h100(
    precision: str = "bf16", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """H100, baseline config."""
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        gpu="h100",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = pretrain_config()
    cfg.mixed_precision = precision_config

    # Apply model-specific settings that were previously passed as constructor args
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend
    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        # Recompute layout based on updated PP/VP sizes
        set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)

    # Disabling to avoid functional errors. TODO: Test with it enabled and keep it enabled if it works.
    cfg.comm_overlap.overlap_grad_reduce = False

    return cfg


# DeepSeek-V3 cuteDSL 3L proxy (vr200) — diagnostic: does the cuteDSL fused grouped-GEMM-GLU
# compile on Rubin sm100f for a plain (non-DSA, non-clamped) DSv3 MoE? Isolates the cuteDSL/cudnn-fe
# path from DSv4-specific code. Same perf knobs as dsv3 gb200 FP8_MX (full-iter CG + cuteDSL), EP8/PP1.
_DEEPSEEK_V3_PROXY_NUM_LAYERS = 3


def deepseek_v3_proxy_pretrain_config_vr200(
    precision: str = "fp8_mx", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """VR200, 3-layer all-MoE DeepSeek-V3 proxy (MXFP8 + cuteDSL) to test grouped-GEMM-GLU on Rubin."""
    if precision != "fp8_mx":
        raise NotImplementedError(f"dsv3 vr200 proxy supports precision='fp8_mx' only; got {precision!r}.")
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3_proxy",
        gpu="vr200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )
    cfg = pretrain_config()
    cfg.mixed_precision = get_precision_config(precision)
    if cfg.mixed_precision.fp8_recipe == "mxfp8":
        cfg.model.fp8_output_proj = True

    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.expert_model_parallel_size = base_cfg.expert_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    # 3 ALL-MoE layers (DSv3's first_k_dense_replace would otherwise make a 3L proxy all-dense and
    # never hit the MoE grouped GEMM). Drop MTP. PP1 needs no explicit pipeline layout.
    n = _DEEPSEEK_V3_PROXY_NUM_LAYERS
    cfg.model.num_layers = n
    cfg.model.mtp_num_layers = None
    if hasattr(cfg.model, "moe_layer_freq"):
        cfg.model.moe_layer_freq = [1] * n
    if hasattr(cfg.model, "first_k_dense_replace"):
        cfg.model.first_k_dense_replace = 0

    # PP1: no pipeline split, so no custom layout needed. Just CLEAR the base recipe's stale layout
    # (it still encodes 61 decoder layers -> "Number of decoder layers 61 must match num_layers 3").
    cfg.model.pipeline_model_parallel_layout = None

    set_deepseek_v3_common_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)
    cfg.comm_overlap.overlap_grad_reduce = True
    return cfg


# Number of decoder layers in the DeepSeek-V4-Pro proxy. The proxy is a
# pipeclean stand-in for the full 61-layer Pro model: it keeps every production
# Pro hyperparameter (experts, MLA LoRA ranks, hash routing, SwiGLU clamp,
# DSA/mHC) but shrinks the layer count so it fits on a single NVL72 domain.
_DEEPSEEK_V4_PRO_PROXY_NUM_LAYERS = 8


def set_deepseek_v4_pro_common_configs(cfg: ConfigContainer) -> None:
    """Set the performance knobs shared by full DeepSeek-V4-Pro and its proxy.

    Reproduces the validated full-iteration CUDA-graph + fused-DSA recipe
    (calibrated ~917 TFLOP/s/GPU on the 8L proxy, GB200). Call AFTER
    ``set_workload_base_configs``/``set_full_iter_cg_configs`` so these win over
    ``_set_common_perf_overrides`` (which forces the TE op fuser off and the
    cross-entropy fusion impl back to ``te``).
    """
    cfg.model.moe_router_force_load_balancing = True

    # Fused DSA sparse attention (FlashMLA forward + cuDNN DSA backward) — the
    # dominant perf lever over the unfused PyTorch DSA path.
    cfg.model.apply_dsa_kernel_fusion = True

    # TE op fuser + native cross-entropy fusion (dev backbone disables the "te"
    # cross-entropy fusion path for DSv4).
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # cuteDSL fused grouped-MLP interleave (clamped SwiGLU fusion path).
    cfg.model.moe_mlp_glu_interleave_size = 32

    # Native global MXFP8 (matching the MLM reference), overriding the lib mxfp8 config's
    # eval-oriented choices. The lib bundles a per-layer "kitchen" quant_recipe
    # (TEQuantizationParams, MXFP8-train/BF16-eval) with fp8_param_gather=False; that exists
    # only for DSv4 MTP/validation BF16 eval, which a perf benchmark (eval_iters=0) doesn't
    # use. Perf wants standard TE MXFP8 with fp8 param gather ON (perf-optimal, = MLM).
    cfg.model.quant_recipe = None
    cfg.mixed_precision.fp8_param_gather = True
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = True

    # Train the DSA/CSA indexer, matching the MLM reference. The lib's DSv4-Flash
    # recipe zeroes the indexer auxiliary loss (eval-leaning, same commit as the
    # kitchen quant_recipe above), which leaves the sparse-token selector without a
    # direct training signal. A faithful pretraining mirror keeps it on.
    cfg.model.dsa_indexer_loss_coeff = 0.01
    cfg.model.dsa_indexer_use_sparse_loss = True

    # No CPU (pinned host) paged-stash spill buffer, matching the MLM reference
    # (cpu factor 0.0; cuda factor stays 1.2). set_full_iter_cg_configs defaults
    # this to 1.0, which mirrors the multi-tens-of-GB stash working set into
    # page-locked host RAM per rank -- with 4 ranks/GB300 node that blows the
    # host cgroup (OOM-killed at iter 2). The 1.2x HBM buffer holds the stash alone.
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 0.0

    # Log GPU memory every log_interval steps (mirrors MLM's --log-memory-interval),
    # so steady-state peak memory is captured. MBridge's flag-gated report otherwise
    # stops after iter 2, which under full-iter CG undercounts the post-capture peak.
    cfg.logger.log_memory_interval = cfg.logger.log_interval

    # BF16 precision-aware optimizer master gradients, matching the MLM reference.
    # The lib forces main_grads_dtype=fp32 (deepseek_v4.py); MLM uses bf16. With the
    # precision-aware optimizer (enabled here) bf16 master grads are valid and halve
    # the master-grad buffer. grad_reduce_in_fp32 (the DDP reduce path) is already
    # False above -- this is the separate optimizer-side knob.
    cfg.optimizer.main_grads_dtype = torch.bfloat16

    # MCore's TransformerConfig.__post_init__ does set(self.offload_modules);
    # keep it an empty list (not None) when fine-grained offloading is off. The
    # full-Pro builder overrides this with the real offload module list.
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = []

    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False


def _deepseek_v4_pro_pretrain_config(
    gpu: str,
    precision: str,
    config_variant: str,
    *,
    proxy: bool,
    proxy_num_layers: int = _DEEPSEEK_V4_PRO_PROXY_NUM_LAYERS,
) -> ConfigContainer:
    """Build a DeepSeek-V4-Pro (or proxy) performance config for ``gpu``.

    When ``proxy`` is True the full 61L Pro is shrunk to ``proxy_num_layers`` decoder
    layers (default 8 for the GB200/GB300 single-NVL72 proxy; the Rubin pipeclean uses 3).
    """
    if precision != "fp8_mx":
        raise NotImplementedError(
            "DeepSeek-V4-Pro performance configs currently support precision='fp8_mx' "
            f"(MXFP8) only; got {precision!r}."
        )

    model_recipe_name = "deepseek_v4_pro_proxy" if proxy else "deepseek_v4_pro"
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name=model_recipe_name,
        gpu=gpu,
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )

    cfg = deepseek_v4_pro_pretrain_mxfp8_config()

    # Pre-apply parallelism so the pipeline-layout helper sees the right PP/VPP.
    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.expert_model_parallel_size = base_cfg.expert_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    if proxy:
        # Shrink the 61L Pro to an 8L proxy: drop MTP and truncate the per-layer
        # lists the bridge derives from num_hidden_layers=61 to the proxy length.
        # csa_compress_ratios is 62 (61 layers + 1 MTP); moe_layer_freq is 61
        # (all-MoE). Both must equal num_layers after the override.
        n = proxy_num_layers
        cfg.model.num_layers = n
        cfg.model.mtp_num_layers = None
        cfg.model.csa_compress_ratios = list(cfg.model.csa_compress_ratios)[:n]
        if isinstance(cfg.model.moe_layer_freq, list):
            cfg.model.moe_layer_freq = cfg.model.moe_layer_freq[:n]

    if base_cfg.pp_layout:
        cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout
    else:
        set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)
    set_deepseek_v4_pro_common_configs(cfg)

    if not proxy:
        # Full Pro: fine-grained activation offloading of attention activations
        # (matches the validated 61L GB300 full-iter-CG run).
        cfg.model.fine_grained_activation_offloading = True
        cfg.model.offload_modules = ["core_attn", "attn_proj"]
        cfg.model.fine_grained_offloading_max_inflight_offloads = 2

    cfg.comm_overlap.overlap_grad_reduce = True

    return cfg


def deepseek_v4_pro_pretrain_config_gb200(
    precision: str = "fp8_mx", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, full 61-layer DeepSeek-V4-Pro perf config (MXFP8)."""
    return _deepseek_v4_pro_pretrain_config(
        gpu="gb200", precision=precision, config_variant=config_variant, proxy=False
    )


def deepseek_v4_pro_pretrain_config_gb300(
    precision: str = "fp8_mx", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, full 61-layer DeepSeek-V4-Pro perf config (MXFP8)."""
    return _deepseek_v4_pro_pretrain_config(
        gpu="gb300", precision=precision, config_variant=config_variant, proxy=False
    )


def deepseek_v4_pro_pretrain_config_vr200(
    precision: str = "fp8_mx", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """VR200 (Rubin), full 61-layer DeepSeek-V4-Pro perf config (MXFP8).

    Identical workload/parallelism to the GB300 full Pro (TP1/PP4/VPP4/EP64, GBS4096,
    256 GPUs, MTP=1, layout ``Et*4|(tttt|)*14tmL``, recompute mla_up_proj+mhc, fine-grained
    activation offloading of core_attn/attn_proj) via DEEPSEEK_V4_PRO_PRETRAIN_CONFIG_VR200_FP8_MX_V1
    (= the GB300 base config), plus the Rubin-only SwiGLU-clamp fix that the proxy uses.
    NVTE_CPU_OFFLOAD_V1=1 (FGO) and the NCCL GDR env are supplied by the launcher.
    """
    cfg = _deepseek_v4_pro_pretrain_config(
        gpu="vr200", precision=precision, config_variant=config_variant, proxy=False
    )
    # The Rubin sm100 cuteDSL grouped-GEMM-GLU kernel rejects clamped SwiGLU tuning. DSv4 sets
    # activation_func_clamp_value=10.0 (from HF swiglu_limit); disabling it (None) makes the fused
    # GLU path run. Same vr200-only fix as deepseek_v4_pro_proxy_pretrain_config_vr200; Blackwell
    # keeps the clamp. Revert once the Rubin grouped-GEMM-GLU kernel supports clamped SwiGLU.
    cfg.model.activation_func_clamp_value = None
    return cfg


def deepseek_v4_pro_proxy_pretrain_config_gb200(
    precision: str = "fp8_mx", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, 8-layer DeepSeek-V4-Pro proxy perf config (MXFP8, pipeclean)."""
    return _deepseek_v4_pro_pretrain_config(
        gpu="gb200", precision=precision, config_variant=config_variant, proxy=True
    )


def deepseek_v4_pro_proxy_pretrain_config_gb300(
    precision: str = "fp8_mx", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB300, 8-layer DeepSeek-V4-Pro proxy perf config (MXFP8, pipeclean)."""
    return _deepseek_v4_pro_pretrain_config(
        gpu="gb300", precision=precision, config_variant=config_variant, proxy=True
    )


# Rubin (VR200) fail-fast pipeclean: 3 layers on 8 GPUs (2 nodes). Smaller than the 8L
# single-NVL72 proxy so the full MXFP8 + full-iter-CG + fused-DSA stack can be brought up
# and iterated on quickly during Rubin bring-up.
_DEEPSEEK_V4_PRO_PROXY_VR200_NUM_LAYERS = 3
# v2 scale-up: 16 layers in ONE NVL domain (64 GPUs, PP1) to mimic the full Pro's PP-stage-0
# depth (~15-16 layers) WITHOUT any inter-block pipeline comm -- isolates the per-stage cost
# from the cross-NVL-domain PP point-to-point that dominates the 256-GPU full run (job 136601).
_DEEPSEEK_V4_PRO_PROXY_VR200_V2_NUM_LAYERS = 16
# v3 "first+last rank" probe: 29 layers over PP2/VPP4 on 128 GPUs (2 NVL blocks) + MTP=1. The
# pp_layout "Et*4|(t*4|)*6tmL" makes PP rank 0 = embed+16L (= full Pro stage 0) and PP rank 1 =
# 13L+MTP+loss (= full Pro LAST stage), so it reproduces the full model's first AND last ranks
# while adding one cross-NVL-domain PP boundary to measure. Needs MTP + custom pp_layout + FGO,
# so it takes a dedicated branch (like deepseek_v4_pro_proxy_pp2_pretrain_config_gb200), not the
# simple proxy_num_layers path used by v1/v2.
_DEEPSEEK_V4_PRO_PROXY_VR200_V3_NUM_LAYERS = 29


def _deepseek_v4_pro_proxy_vr200_v3(precision: str) -> ConfigContainer:
    """Build the VR200 v3 probe: 29L over PP2/VPP4 + MTP on 128 GPUs (2 NVL blocks).

    Reproduces the full 61L Pro's FIRST and LAST pipeline ranks at half scale: PP rank 0 =
    embed + 16 layers (= full Pro stage 0); PP rank 1 = 13 layers + MTP + loss (= full Pro last
    stage). Adds exactly one cross-NVL-domain PP boundary so its cost can be measured. Carries the
    full Pro's recompute (mla_up_proj+mhc) + fine-grained activation offloading so the heavy stage 0
    fits the POR ceiling, plus the Rubin SwiGLU-clamp fix. NVTE_CPU_OFFLOAD_V1=1 + NCCL GDR env are
    supplied by the launcher.
    """
    if precision != "fp8_mx":
        raise NotImplementedError(
            "DeepSeek-V4-Pro performance configs currently support precision='fp8_mx' "
            f"(MXFP8) only; got {precision!r}."
        )
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v4_pro_proxy",
        gpu="vr200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant="v3",
    )

    cfg = deepseek_v4_pro_pretrain_mxfp8_config()

    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.expert_model_parallel_size = base_cfg.expert_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    # 29 layers, KEEP MTP=1. Truncate the bridge-derived per-layer lists:
    # csa_compress_ratios -> first 29 (real) + [0] (MTP slot) = 30; moe_layer_freq -> first 29 (all-MoE).
    n = _DEEPSEEK_V4_PRO_PROXY_VR200_V3_NUM_LAYERS
    cfg.model.num_layers = n
    cfg.model.mtp_num_layers = 1
    cfg.model.csa_compress_ratios = list(cfg.model.csa_compress_ratios)[:n] + [0]
    if isinstance(cfg.model.moe_layer_freq, list):
        cfg.model.moe_layer_freq = cfg.model.moe_layer_freq[:n]

    # pp_layout DSL from the workload base config (mcore parses it).
    cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout

    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)
    set_deepseek_v4_pro_common_configs(cfg)

    # Fine-grained activation offloading (matches the full 61L Pro) so the heavy stage 0
    # (embed + 16 layers) fits the POR ceiling. Must be set AFTER set_deepseek_v4_pro_common_configs
    # (which forces FGO off). The launcher must pass NVTE_CPU_OFFLOAD_V1=1.
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["core_attn", "attn_proj"]
    cfg.model.fine_grained_offloading_max_inflight_offloads = 2

    # Rubin sm100 cuteDSL grouped-GEMM-GLU kernel rejects clamped SwiGLU (vr200-only fix).
    cfg.model.activation_func_clamp_value = None

    cfg.comm_overlap.overlap_grad_reduce = True
    return cfg


def deepseek_v4_pro_proxy_pretrain_config_vr200(
    precision: str = "fp8_mx", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """VR200 (Rubin) DeepSeek-V4-Pro proxy perf config (MXFP8).

    ``config_variant`` selects the Rubin proxy scale:
      - ``v1`` -> 3 layers on 8 GPUs (TP1/PP1/EP8): fail-fast pipeclean.
      - ``v2`` -> 16 layers on 64 GPUs (TP1/PP1/EP64) WITH recompute (mla_up_proj+mhc, like full
        Pro): single-NVL-domain depth probe, no inter-block PP comm (isolates per-stage cost).
      - ``v3`` -> 29 layers on 128 GPUs (TP1/PP2/VPP4/EP64) + MTP: probe that reproduces the full
        61L Pro's first rank (embed+16L) and last rank (13L+MTP+loss), adding one cross-NVL-domain
        PP boundary.
    """
    if config_variant.lower() == "v3":
        return _deepseek_v4_pro_proxy_vr200_v3(precision)
    proxy_num_layers = (
        _DEEPSEEK_V4_PRO_PROXY_VR200_V2_NUM_LAYERS
        if config_variant.lower() == "v2"
        else _DEEPSEEK_V4_PRO_PROXY_VR200_NUM_LAYERS
    )
    cfg = _deepseek_v4_pro_pretrain_config(
        gpu="vr200",
        precision=precision,
        config_variant=config_variant,
        proxy=True,
        proxy_num_layers=proxy_num_layers,
    )
    # The Rubin sm100 cuteDSL grouped-GEMM-GLU kernel rejects clamped SwiGLU tuning
    # (geglu_alpha / glu_clamp_max / glu_clamp_min). DSv4 sets activation_func_clamp_value=10.0
    # (from HF swiglu_limit); disabling it (None) makes the fused GLU path run, matching dsv3
    # (which leaves clamp at the mcore default None and works with cuteDSL). This drops DSv4's
    # SwiGLU clamp=10 stability feature — acceptable for the Rubin perf pipeclean; revert once the
    # Rubin grouped-GEMM-GLU kernel supports clamped SwiGLU. vr200-only (Blackwell keeps the clamp).
    cfg.model.activation_func_clamp_value = None

    # NOTE: single_grouped_weight (dense grouped-GEMM) is a *workaround* probe for the Rubin cuteDSL
    # `_compile_discrete` tvm-ffi crash — NOT what MLPerf does (its successful DSv3 run uses the default
    # discrete path, moe_single_grouped_weight=False). Toggle via DSV4_RUBIN_SINGLE_GROUPED_WEIGHT=1.
    if os.environ.get("DSV4_RUBIN_SINGLE_GROUPED_WEIGHT") == "1":
        cfg.model.moe_grouped_gemm = True
        cfg.model.moe_single_grouped_weight = True
    return cfg


# Multi-stage debug proxy: PP2/VPP4 + MTP, 15 layers. Used to reproduce the full-Pro
# scaling-mode crash at small scale (the PP1 proxy with MTP runs fine, so the trigger
# needs a multi-stage pipeline + interleaved schedule + MTP on a non-first stage).
_DEEPSEEK_V4_PRO_PROXY_PP2_NUM_LAYERS = 13


def deepseek_v4_pro_proxy_pp2_pretrain_config_gb200(
    precision: str = "fp8_mx", mock: bool = True, config_variant: str = "v1"
) -> ConfigContainer:
    """GB200, 15-layer DeepSeek-V4-Pro proxy with PP2/VPP4 + MTP (MXFP8, 128 GPUs).

    Multi-stage debug variant: mimics the full Pro's pipeline + interleaved schedule +
    MTP-on-last-stage at small scale. EP64 (like full Pro) -> PP2*EP64 = 128 GPUs.
    15 transformer layers over 8 virtual stages (PP2*VPP4); last stage = "tmL" (1 layer
    + MTP + loss). num_layers / CSA ratios / MTP / pp_layout are set HERE in Python (not
    via Hydra overrides, which mis-parse the layout DSL into a single element).
    """
    if precision != "fp8_mx":
        raise NotImplementedError(
            "DeepSeek-V4-Pro performance configs currently support precision='fp8_mx' "
            f"(MXFP8) only; got {precision!r}."
        )
    base_cfg = get_workload_base_config(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v4_pro_proxy_pp2",
        gpu="gb200",
        compute_dtype=precision.upper(),
        task="pretrain",
        config_variant=config_variant,
    )

    cfg = deepseek_v4_pro_pretrain_mxfp8_config()

    cfg.model.pipeline_model_parallel_size = base_cfg.pipeline_model_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = base_cfg.virtual_pipeline_model_parallel_size
    cfg.model.expert_model_parallel_size = base_cfg.expert_model_parallel_size
    cfg.model.moe_flex_dispatcher_backend = base_cfg.moe_flex_dispatcher_backend

    # 15 layers, KEEP MTP=1 (mimic full Pro). Truncate the bridge-derived per-layer lists:
    # csa_compress_ratios -> first 15 (real) + [0] (MTP slot) = 16 == num_layers + mtp;
    # moe_layer_freq -> first 15 (all-MoE).
    n = _DEEPSEEK_V4_PRO_PROXY_PP2_NUM_LAYERS
    cfg.model.num_layers = n
    cfg.model.mtp_num_layers = 1
    cfg.model.csa_compress_ratios = list(cfg.model.csa_compress_ratios)[:n] + [0]
    if isinstance(cfg.model.moe_layer_freq, list):
        cfg.model.moe_layer_freq = cfg.model.moe_layer_freq[:n]

    # pp_layout string from the workload base config, set in Python (mcore parses the DSL).
    cfg.model.pipeline_model_parallel_layout = base_cfg.pp_layout

    set_workload_base_configs(cfg, base_cfg)
    if is_full_iteration_cuda_graph(cfg.model):
        set_full_iter_cg_configs(cfg)
    set_deepseek_v4_pro_common_configs(cfg)

    cfg.comm_overlap.overlap_grad_reduce = True
    return cfg
