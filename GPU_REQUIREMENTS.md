# MLLM-MSR GPU Requirements Estimate

## Project Overview

This project implements **"Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation"** (AAAI-25), using MLLMs for sequential recommendation with multimodal data.

## Models Used

| Model | Parameters | Purpose |
|-------|-----------|---------|
| **LLaVA-v1.6-Mistral-7B** | ~7B | Image captioning + fine-tuned recommendation |
| **Meta-Llama-3-8B-Instruct** | ~8B | User preference inference & summarization |

## Pipeline Stages & GPU Requirements

### Stage 1: Image Summarization (Inference)
- Model: LLaVA-v1.6-Mistral-7B (float16)
- Config: **6 GPUs**, batch_size=8, 4 processes
- Per-GPU VRAM: ~16-20GB
- **Total VRAM: ~96-120GB**

### Stage 2: User Preference Inference (Inference)
- Model: Llama-3-8B-Instruct (bfloat16)
- Config: **6 GPUs**, batch_size=12, 6 processes
- Per-GPU VRAM: ~18-22GB
- **Total VRAM: ~108-132GB**

### Stage 3: Model Training (Most Resource-Intensive)
- Model: LLaVA-v1.6-Mistral-7B + LoRA fine-tuning
- Framework: PyTorch Lightning + DeepSpeed Stage 2
- Config: **6 GPUs**, batch_size=1/GPU, gradient_accumulation=4
- Precision: float16 mixed precision
- Training data: 25,000 samples, 4 epochs
- Per-GPU VRAM: ~20-24GB
- **Total VRAM: ~120-144GB**

### Stage 4: Evaluation (Testing)
- Config: **8 GPUs**, batch_size=6, 6 processes
- Uses Flash Attention 2
- Per-GPU VRAM: ~16-20GB
- **Total VRAM: ~128-160GB**

## VRAM Breakdown (Training, Per GPU)

| Component | VRAM |
|-----------|------|
| Model parameters (float16) | ~14GB |
| LoRA trainable parameters | ~0.5-1GB |
| AdamW optimizer states | ~2-3GB |
| Gradients | ~1GB |
| Activations (batch=1, seq=1024) | ~2-3GB |
| **Total per GPU** | **~20-22GB** |

> DeepSpeed Stage 2 shards gradients and optimizer states across GPUs, potentially reducing per-GPU usage to ~16-18GB.

## Recommended Hardware

### Minimum (Runnable)
- **6× NVIDIA A30 (24GB)** or **6× RTX 3090 (24GB)**
- Total VRAM: 144GB
- Requires Flash Attention 2 support (Ampere architecture+)

### Recommended (Comfortable)
- **6-8× NVIDIA A100 (40GB/80GB)**
- Total VRAM: 240-640GB
- Allows larger batch sizes for faster training

### Budget (With Trade-offs)
- **4× RTX 4090 (24GB)**
- Requires modifying GPU count configs
- May need QLoRA (4-bit quantization, already supported in code)
- Inference stages run sequentially

## Technical Requirements

| Requirement | Details |
|-------------|---------|
| **CUDA** | Required, CPU-only not supported |
| **Flash Attention 2** | Mandatory, needs Ampere (A100/A30/RTX 30xx) or newer |
| **DeepSpeed** | Required for distributed training |
| **NCCL** | Multi-GPU communication backend |
| **Disk Space** | ~30GB model weights + dataset + checkpoints, recommend **100GB+** |

## Summary

| Item | Value |
|------|-------|
| **Minimum GPU Count** | 6 (default config) |
| **Minimum Per-GPU VRAM** | 24GB |
| **Minimum Total VRAM** | ~144GB |
| **Recommended Total VRAM** | 240GB+ |
| **Recommended GPU** | A100-40GB/80GB × 6-8 |
| **Minimum Viable GPU** | RTX 3090/A30 24GB × 6 |
