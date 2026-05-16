# festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8 on 8-node DGX Spark sm_121

vLLM patches and recipe to deploy [festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8](https://huggingface.co/festr2/MiMo-V2.5-Pro-NVFP4-MXFP8-attn-TP8) on an 8-node DGX Spark cluster (GB10 sm_121 silicon, 121 GiB unified memory per node) via Ray-distributed vLLM with TP=8.

Coherent output confirmed 2026-05-16. Chat completion `"Say hello"` returns `"Hey there! 👋 How's it going?"`; `"12 × 7"` returns `"84"`; `def fibonacci(n):` continues with proper Python recursion.

## What's in here

| File | Purpose |
|---|---|
| `patch_marlin_chunked_repack.py` | Stream Marlin NVFP4 MoE per-expert repack into a preallocated output tensor instead of accumulating + `torch.cat`. Saves ~36 GiB peak head memory and avoids head-node OOM. Targets `prepare_nvfp4_moe_layer_for_marlin` in `marlin_utils_fp4.py`. |
| `patch_modelopt_mxfp8_dispatch.py` | Add an MXFP8 branch to `ModelOptMixedPrecisionConfig.get_quant_method` in `modelopt.py`. Without this, festr2's MXFP8 attention layers (`quant_algo: MXFP8`) fall through to `UnquantizedLinearMethod` and produce garbage. |
| `patch_modelopt_mxfp8_scale_name.py` | Rename `weight_scale_inv` → `weight_scale` in `ModelOptMxFp8LinearMethod.process_weights_after_loading`. festr2 stores the scale under the `_inv`-suffixed key (modelopt convention), but the byte values are *forward* E8M0 — **no byte-flip needed** (`254 - byte` makes it worse). |
| `patch_mimo_qkv_split.py` | **The real fix.** Replace vLLM's `mimo_v2.py` naive `loaded_weight.chunk(tp_size, dim=0)[tp_rank]` for fused `qkv_proj` with proper `[Q\|K\|V]` slicing + 3× `QKVParallelLinear.weight_loader` calls with `shard_id` `"q"`/`"k"`/`"v"`. Prior code gave 7 of 8 ranks all-Q-region rows in their K/V parameter slots → 7 of 8 ranks computed attention with Q values posing as K and V. Confirmed empirically via E8M0 scale-byte distribution: Q section mean 114.55, K 115.99, V 113.27 (clear Q→K→V layout). |
| `recipes/4x-spark-cluster/mimo-v2.5-pro-c25.yaml` | Recipe for the [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) `run-recipe.py` wrapper. Uses FlashInferCutlassMxfp8 attention + Marlin NVFP4 MoE + FP8 KV cache + `triton_attn_diffkv` + `enforce_eager` + `--moe-backend marlin`. |
| `launch-cluster.sh` | The base eugr launcher with all 4 patches wired in (search for `MIMO-` markers). Each patch runs inside every node's container at start. |

## Usage

```bash
# From the launcher host (typically AO1 for an 8-node Ray cluster):
cd ~/spark-vllm-docker-main
export VLLM_SPARK_EXTRA_DOCKER_ARGS="--tmpfs /tmp/ray:exec,size=4g"
nohup bash run-recipe.sh recipes/4x-spark-cluster/mimo-v2.5-pro-c25.yaml \
  -n 192.168.0.11,192.168.0.10,192.168.0.12,192.168.0.13,192.168.0.14,192.168.0.15,192.168.0.16,192.168.0.17 \
  -e RAY_agent_register_timeout_ms=900000 \
  -e RAY_health_check_initial_delay_ms=900000 \
  -e RAY_health_check_timeout_ms=180000 \
  -e RAY_health_check_period_ms=10000 \
  -e RAY_health_check_failure_threshold=2000 \
  -e RAY_grpc_keepalive_time_ms=60000 \
  -e RAY_grpc_keepalive_timeout_ms=600000 \
  -e VLLM_RPC_TIMEOUT=14400000 \
  >/tmp/mimo.log 2>&1 &
```

The first IP in `-n` is the Ray head. Endpoint comes up on `http://<head>:5001/v1/...`.

Boot takes ~30 min on a healthy cluster, dominated by the head node's per-rank weight load (head competes for unified memory with Ray master + EngineCore + APIServer). If the head node is memory-thrashing (`free -h` shows <2 GiB free + UVM kthread at high CPU), boot can stretch to 2 h.

## Bug story

Original hypothesis (May 12-13) was a CUTLASS Sm120 ArchTag bug — the garbage signature `あるい...` matched flashinfer #2776. We declared a hard blocker and waited for upstream.

May 15-16: re-investigation found that

1. C13/14 garbage `あるい` was MXFP8 attention falling through to `UnquantizedLinearMethod` (no quant handler matched `MXFP8` quant_algo).
2. After wiring MXFP8 dispatch (`patch_modelopt_mxfp8_dispatch.py`), garbage changed to `ייך` because the scale parameter was registered under `weight_scale` but festr2's checkpoint key is `weight_scale_inv` — loader couldn't match by name, so the param stayed zero-filled.
3. After fixing the rename (`patch_modelopt_mxfp8_scale_name.py`), garbage changed to `*\0` because vLLM's `mimo_v2.py` fused-qkv loader did naive `chunk(tp_size, dim=0)[rank]` on the `[Q|K|V]` tensor.
4. Once the QKV split was fixed (`patch_mimo_qkv_split.py`), the model produces coherent output.

The CUTLASS Sm120 hypothesis was wrong. The bug was always in vLLM's load path.

## License

Patches are MIT-licensed; the launcher and recipes come from [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (also MIT).
