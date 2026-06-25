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

"""Workload base presets for DeepSeek-V3 performance configs.

Config naming convention:
    {MODEL}_{SIZE}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

V1: GBS=2048 for Blackwell variants, GBS=8192 for H100
V2: GBS=4096 for Blackwell variants, GBS=16384 for H100

Use --config_variant to select a variant.
Use --list_config_variants to see available variants interactively.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_DEEPSEEK_V3_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
)


# =============================================================================
# DeepSeek V3 Pretrain - V1 (original GBS settings)
# =============================================================================

DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1 = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=256,
    global_batch_size=2048,
    micro_batch_size=2,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    pp_layout="Et*4|(t*4|)*14tmL",
    expert_model_parallel_size=32,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    cuda_graph_scope=[],
    recompute_modules=["mla_up_proj"],
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16_V1 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1,
    micro_batch_size=1,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
    recompute_modules=["moe_act"],
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_CS_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1,
    micro_batch_size=1,
    cuda_graph_impl="full_iteration",
    moe_a2a_overlap=True,
    cutedsl_fused_grouped_mlp=True,
    fp8_dot_product_attention=True,
    recompute_modules=[],
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1


DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1 = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=256,
    global_batch_size=2048,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    recompute_modules=["mlp"],
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_BF16_V1 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1,
    recompute_modules=["mla_up_proj"],
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_CS_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1,
    cuda_graph_impl="full_iteration",
    cuda_graph_scope=[],
    moe_a2a_overlap=True,
    cutedsl_fused_grouped_mlp=True,
    recompute_modules=["mla_up_proj"],
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1


DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_V1 = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=128,
    global_batch_size=4096,
    micro_batch_size=1,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=8,
    pp_layout="Et*4|(t*4|)*14tmL",
    expert_model_parallel_size=64,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
    recompute_modules=["mla_up_proj"],
)
DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_BF16_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_FP8_CS_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_FP8_MX_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_NVFP4_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_V1


DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V1 = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=256,
    pipeline_model_parallel_size=16,
    expert_model_parallel_size=8,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj"],
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_BF16_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_CS_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_CS_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_NVFP4_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V1


DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V1 = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=256,
    pipeline_model_parallel_size=16,
    expert_model_parallel_size=8,
    global_batch_size=2048,
    recompute_modules=["mla_up_proj"],
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_BF16_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_CS_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_CS_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_NVFP4_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V1


DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V1 = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=1024,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    global_batch_size=8192,
    recompute_modules=["mla_up_proj", "mlp"],
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    pp_layout="Et|(tt|)*30mL",
)
DEEPSEEK_V3_PRETRAIN_CONFIG_H100_BF16_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_CS_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V1
DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_V1 = DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_CS_V1


# =============================================================================
# DeepSeek V3 Pretrain - V2 (GBS=4096 for Blackwell, GBS=16384 for H100)
# =============================================================================

DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V1,
    global_batch_size=4096,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16_V1,
    global_batch_size=4096,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_CS_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V1,
    global_batch_size=4096,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_V2


DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V1,
    global_batch_size=4096,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_BF16_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_BF16_V1,
    global_batch_size=4096,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_CS_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_V1,
    global_batch_size=4096,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_V2


DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_BF16_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_BF16_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_FP8_CS_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_CS_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_FP8_MX_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_NVFP4_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V2


DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V1,
    global_batch_size=4096,
    micro_batch_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=None,
    recompute_modules=["mla_up_proj"],
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)

DEEPSEEK_V3_PRETRAIN_CONFIG_B300_BF16_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_CS_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_BF16_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_BF16_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_B300_NVFP4_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B300_BF16_V2


DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V1,
    global_batch_size=4096,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_BF16_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_CS_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_CS_V2,
    moe_flex_dispatcher_backend="deepep",
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=2,
    expert_model_parallel_size=32,
    moe_a2a_overlap=True,
    pp_layout="Et*4|(t*4|)*14tmL",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
    recompute_modules=["mla_up_proj", "mlp"],
)
DEEPSEEK_V3_PRETRAIN_CONFIG_B200_NVFP4_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_B200_V2,
    moe_flex_dispatcher_backend="deepep",
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=2,
    expert_model_parallel_size=32,
    moe_a2a_overlap=True,
    pp_layout="Et*4|(t*4|)*14tmL",
    cuda_graph_impl="none",
    recompute_modules=["mla_up_proj", "layernorm", "moe_act"],
)


DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V1,
    global_batch_size=16384,
)
DEEPSEEK_V3_PRETRAIN_CONFIG_H100_BF16_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_CS_V2 = DEEPSEEK_V3_PRETRAIN_CONFIG_H100_V2
DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_V2 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_CS_V2,
    virtual_pipeline_model_parallel_size=2,
    pp_layout=None,
)


# =============================================================================
# DeepSeek V3 Pretrain - Large Scale Proxy
# =============================================================================

DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_LARGE_SCALE = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16_V1,
    global_batch_size=256,
)


DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_LARGE_SCALE = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_CS_V1,
    global_batch_size=256,
)


DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_LARGE_SCALE = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_V1,
    global_batch_size=256,
)


DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_LARGE_SCALE = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_V1,
    global_batch_size=256,
)


DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_LARGE_SCALE = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_V1,
    global_batch_size=1024,
    virtual_pipeline_model_parallel_size=2,
    pp_layout=None,
)


# =============================================================================
# DeepSeek V3 Pretrain - GBS15360 (GB300 MXFP8, GBS=15*1024)
# =============================================================================

DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_GBS15360 = replace(
    DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V2,
    global_batch_size=15360,
)


# =============================================================================
# DeepSeek V4 Pro Pretrain (MXFP8, full-iteration CUDA graph + fused DSA)
#
# Full Pro is the 61-layer production model. The proxy shrinks it to 8 layers
# for single-NVL72-domain pipecleans. Layer count / CSA compress ratios / MTP
# are overridden in the perf builder (deepseek_llm_pretrain.py), not here -
# WorkloadBaseConfig has no num_layers field. Mirrors the dsv3 FP8_MX full-iter
# pattern: cuda_graph_impl="full_iteration" + cutedsl_fused_grouped_mlp=True
# auto-plumb the cuteDSL/CG env vars via perf_plugins.
# =============================================================================

BASE_DEEPSEEK_V4_PRO_CONFIG = WorkloadBaseConfig(
    expert_tensor_parallel_size=1,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    cuda_graph_impl="full_iteration",
    cuda_graph_scope=[],
    cutedsl_fused_grouped_mlp=True,
)

# Full 61-layer Pro: TP=1, PP=4, VPP=4, EP=64 on 256 GPUs, GBS=4096.
DEEPSEEK_V4_PRO_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = replace(
    BASE_DEEPSEEK_V4_PRO_CONFIG,
    num_gpus=256,
    micro_batch_size=1,
    global_batch_size=4096,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    pp_layout="Et*4|(tttt|)*14tmL",
    recompute_modules=["mla_up_proj", "mhc"],
)
DEEPSEEK_V4_PRO_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = DEEPSEEK_V4_PRO_PRETRAIN_CONFIG_GB300_FP8_MX_V1

# Rubin (VR200) full 61-layer Pro: identical workload to the GB300 full Pro (TP1/PP4/VPP4/EP64,
# GBS4096, 256 GPUs, FGO of core_attn/attn_proj). The vr200 builder adds the Rubin SwiGLU-clamp
# fix; NVTE_CPU_OFFLOAD_V1 (FGO) + NCCL GDR env are passed by the launcher (not auto-set for vr200).
DEEPSEEK_V4_PRO_PRETRAIN_CONFIG_VR200_FP8_MX_V1 = DEEPSEEK_V4_PRO_PRETRAIN_CONFIG_GB300_FP8_MX_V1

# 8-layer proxy: TP=1, PP=1, EP=64 on 64 GPUs (one NVL72 domain), GBS=2048,
# no recompute (everything fits under full-iteration CUDA graph).
DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = replace(
    BASE_DEEPSEEK_V4_PRO_CONFIG,
    num_gpus=64,
    micro_batch_size=1,
    global_batch_size=2048,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=64,
    pp_layout=None,
    recompute_modules=[],
)
DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_GB200_FP8_MX_V1

# Rubin (VR200) fail-fast pipeclean: 3-layer proxy on 8 GPUs (2 nodes / 4 GPUs each),
# TP1/PP1/EP8, full MXFP8 + full-iteration CUDA-graph + fused-DSA + cuteDSL stack
# (inherits BASE_DEEPSEEK_V4_PRO_CONFIG). EP64 won't fit on 8 GPUs, so EP=8 spans every
# GPU (expert-DP=1); num_layers=3 is applied in the perf builder. GBS=64 -> 8 microbatches
# per DP rank. Smallest shape that still exercises the production perf knobs end-to-end.
DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_VR200_FP8_MX_V1 = replace(
    BASE_DEEPSEEK_V4_PRO_CONFIG,
    num_gpus=8,
    micro_batch_size=1,
    global_batch_size=64,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=8,
    pp_layout=None,
    recompute_modules=[],
)

# Rubin (VR200) single-NVL-domain depth probe (config_variant="v2"): 16 layers on 64 GPUs
# (TP1/PP1/EP64, GBS2048), mirroring the GB200 proxy parallelism but WITH recompute
# (mla_up_proj+mhc, like the full Pro) for memory headroom / safety. PP1 -> no inter-block PP comm,
# so this is the single-domain baseline at full-Pro stage-0 depth. (num_layers=16 set in the builder.)
DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_VR200_FP8_MX_V2 = replace(
    DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_GB200_FP8_MX_V1,
    recompute_modules=["mla_up_proj", "mhc"],
)

# Rubin (VR200) "first+last rank" probe (config_variant="v3"): 29 transformer layers over PP2/VPP4
# on 128 GPUs (2 NVL blocks), EP64, GBS4096, MTP=1. The layout "Et*4|(t*4|)*6tmL" reproduces the
# full 61L Pro's first rank (embed+16L) and last rank (13L+MTP+loss), adding one cross-NVL-domain PP
# boundary so its point-to-point cost can be measured at half scale. recompute mla_up_proj+mhc here
# + FGO in the builder so the heavy stage 0 fits the POR ceiling, matching the full run.
DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_VR200_FP8_MX_V3 = replace(
    BASE_DEEPSEEK_V4_PRO_CONFIG,
    num_gpus=128,
    micro_batch_size=1,
    global_batch_size=4096,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=64,
    pp_layout="Et*4|(t*4|)*6tmL",
    recompute_modules=["mla_up_proj", "mhc"],
)


# DeepSeek-V3 cuteDSL 3L proxy (vr200): diagnostic to test whether the cuteDSL fused
# grouped-GEMM-GLU compiles on Rubin sm100f for a (non-DSA, non-clamped) DSv3 MoE — isolates
# the cuteDSL/cudnn-fe path from DSv4-specific code. 3 all-MoE layers, EP8/PP1 on 8 GPUs,
# full-iteration CG + cuteDSL on (same perf knobs as the dsv3 gb200 FP8_MX config).
DEEPSEEK_V3_PROXY_PRETRAIN_CONFIG_VR200_FP8_MX_V1 = replace(
    BASE_DEEPSEEK_V3_CONFIG,
    num_gpus=8,
    micro_batch_size=1,
    global_batch_size=64,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="hybridep",
    moe_a2a_overlap=False,
    cuda_graph_impl="full_iteration",
    cuda_graph_scope=[],
    cutedsl_fused_grouped_mlp=True,
    pp_layout=None,
    recompute_modules=[],
)

# Multi-stage proxy (PP2/VPP4 + MTP) for reproducing the full-Pro scaling-mode crash
# at small scale. 15 transformer layers over 8 virtual stages (PP2*VPP4); last stage =
# "tmL" (1 layer + MTP + loss), leaner for perf. EP64 mimics full Pro, so this needs
# PP2*EP64 = 128 GPUs.
DEEPSEEK_V4_PRO_PROXY_PP2_PRETRAIN_CONFIG_GB200_FP8_MX_V1 = replace(
    BASE_DEEPSEEK_V4_PRO_CONFIG,
    num_gpus=128,
    micro_batch_size=1,
    global_batch_size=2048,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=2,
    expert_model_parallel_size=64,
    pp_layout="Et*4|t*4|t*4|tmL",
    recompute_modules=[],
)
DEEPSEEK_V4_PRO_PROXY_PP2_PRETRAIN_CONFIG_GB300_FP8_MX_V1 = DEEPSEEK_V4_PRO_PROXY_PP2_PRETRAIN_CONFIG_GB200_FP8_MX_V1


__all__ = [
    # DeepSeek V4 Pro (MXFP8)
    "DEEPSEEK_V4_PRO_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "DEEPSEEK_V4_PRO_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "DEEPSEEK_V4_PRO_PRETRAIN_CONFIG_VR200_FP8_MX_V1",
    "DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_VR200_FP8_MX_V1",
    "DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_VR200_FP8_MX_V2",
    "DEEPSEEK_V4_PRO_PROXY_PRETRAIN_CONFIG_VR200_FP8_MX_V3",
    "DEEPSEEK_V3_PROXY_PRETRAIN_CONFIG_VR200_FP8_MX_V1",
    "DEEPSEEK_V4_PRO_PROXY_PP2_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "DEEPSEEK_V4_PRO_PROXY_PP2_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    # V1 (original GBS settings)
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_CS_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_BF16_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_CS_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_BF16_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_CS_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_NVFP4_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_BF16_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_CS_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_NVFP4_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_BF16_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_FP8_CS_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_FP8_MX_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_NVFP4_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_BF16_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_CS_V1",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_V1",
    # V2 (GBS=4096 for Blackwell, GBS=16384 for H100)
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_BF16_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_CS_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_NVFP4_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_BF16_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_CS_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_NVFP4_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_BF16_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_CS_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_NVFP4_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_BF16_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_CS_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_NVFP4_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_BF16_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_CS_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_BF16_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_FP8_CS_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_FP8_MX_V2",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_VR200_NVFP4_V2",
    # Large Scale Proxy
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_LARGE_SCALE",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB200_FP8_MX_LARGE_SCALE",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B300_FP8_MX_LARGE_SCALE",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_B200_FP8_MX_LARGE_SCALE",
    "DEEPSEEK_V3_PRETRAIN_CONFIG_H100_FP8_SC_LARGE_SCALE",
    # GBS15360 (GB300 MXFP8, GBS=15*1024)
    "DEEPSEEK_V3_PRETRAIN_CONFIG_GB300_FP8_MX_GBS15360",
]
