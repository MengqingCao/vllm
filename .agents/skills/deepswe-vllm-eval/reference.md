# Reference

Read this after the engine is chosen, when presenting the pre-start config,
rendering Pier YAML, or a vLLM source build fails.

## Shared defaults (MiniMax-M3.1-Flash DeepSWE v1.1)

Same for vLLM and SGLang unless the user replaces a line.

| Piece | Default |
| --- | --- |
| Model | `MiniMaxAI/MiniMax-M3.1-Flash` revision `ce7f4df87d2e05262f5775f373dac4772f134041` (252.45 GB, draft head under `dspark/`) |
| HF cache | mount `/SFS-aGqda6ct`, `HF_HUB_CACHE=/SFS-aGqda6ct/huggingface/mencao/hub`, `HF_HUB_OFFLINE=1`. Shared `/SFS-aGqda6ct/huggingface/hub` is not writable. Ask the user for their own cache path. |
| DeepSWE | `https://github.com/datacurve-ai/deep-swe` commit `0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea` (113 `task.toml`). HF `datacurve/deep-swe` is gated; do not use it unless the user says so. |
| Pier | `datacurve-pier==0.3.1` (`uv tool install`). v1.1 grading needs Pier > 0.3.0. |
| Agent | `mini-swe-agent` 2.4.5 plus sandbox package `mcp` |
| Task images | `public.ecr.aws/d3j8x8q7/swe-bench-202605:<id>-v1.1`, amd64, pulled on demand |
| Sampling | `temperature=1.0`, `top_p=0.95`, `reasoning_effort=max` via `chat_template_kwargs` |
| Job | 113 tasks, `n_attempts=2`, `n_concurrent_trials=32`, `environment.delete=true` |
| Per task | 2 CPUs, 8 GB RAM, 20 GB storage, agent timeout 10800s, verifier timeout 1800s, `network_mode=no-network` |
| Listen | `127.0.0.1:8000` on the GB300 |

InferenceX agentic throughput is a different dataset. Do not point Pier at
`semianalysisai/cc-traces-weka-062126`.

## vLLM

Base image `lmsysorg/sglang:dev-minimax-m31-flash` supplies CUDA and PyTorch.
It is not the server. Install `https://github.com/vllm-project/vllm.git` ref
`main` with `tools/use_existing_torch.py`, then
`pip install -r requirements/build/cuda.txt` and
`pip install --no-build-isolation -e .`. `INSTALL_VLLM=1`,
`INSTALL_BUILD_DEPS=1` (needs `libdw-dev`; GCC >= 13). Do not serve this
default from `vllm/vllm-openai:minimax-m31-flash-arm64-cu134` or
`vllm/vllm-openai:v0.33.0`. Record SHA at `$LOG_DIR/vllm-sha.txt`.

`serve.sh` env: `HF_CACHE_MOUNT=/SFS-aGqda6ct`, `HF_HUB_OFFLINE=1`,
`VLLM_ENGINE_READY_TIMEOUT_S=7200`.

Launch with `python3 -m vllm.entrypoints.cli.main serve`. Flags:

```text
--host 127.0.0.1 --port 8000
--served-model-name MiniMaxAI/MiniMax-M3.1-Flash
--language-model-only --trust-remote-code
--reasoning-parser minimax_m3 --tool-call-parser minimax_m3
--enable-auto-tool-choice --block-size 128
--kv-cache-dtype nvfp4
--attention-config {"indexer_kv_dtype":"nvfp4","use_prefill_query_quantization":true}
--disable-hybrid-kv-cache-manager --enable-prefix-caching
--max-model-len 1048576 --max-num-batched-tokens 8192
--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE"}
--disable-uvicorn-access-log --tensor-parallel-size 4 --max-num-seqs 64
--speculative-config {"method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_TS","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"block"}
```

Parser names use underscores (`minimax_m3`). Parallelism is
`--tensor-parallel-size`. KV dtype `nvfp4` is a vLLM value.

## SGLang

Use `lmsysorg/sglang:dev-minimax-m31-flash` as the server. `INSTALL_VLLM=0`.
Do not install vLLM into it and do not replace its PyTorch. Record
`python3 -m sglang.launch_server --version` or the image id in
`$LOG_DIR/sglang-version.txt`.

GB300 balanced recipe from the SGLang MiniMax-M3.1 cookbook. Launch with
`python3 -m sglang.launch_server`. Export `FMHA_SM100_ALLOW_JIT=1` in
`run-serve.sh` before the server. Flags:

```text
--model-path MiniMaxAI/MiniMax-M3.1-Flash
--trust-remote-code
--reasoning-parser minimax-m3 --tool-call-parser minimax-m3
--tp 4
--moe-runner-backend flashinfer_trtllm_routed
--speculative-algorithm DSPARK
--speculative-use-block-verification
--cuda-graph-backend-prefill breakable
--chunked-prefill-size 32768
--max-running-requests 128
--cuda-graph-max-bs-decode 128
--mem-fraction-static 0.8
--tokenizer-worker-num 8
--host 127.0.0.1 --port 8000
```

Parser names use hyphens (`minimax-m3`). Parallelism is `--tp`. There is no
`--kv-cache-dtype nvfp4` and no `--speculative-config` on this command. The
DSpark draft is the checkpoint's `dspark/` folder; do not pass a draft path.
`--max-running-requests` is SGLang's concurrency knob.

The user can ask for the cookbook's low-latency or high-throughput recipe
instead. Low-latency adds the MonoKernel (`--monokernel-backend msa`) and
caps `--max-running-requests 8`. High-throughput adds `--ep 4` and
`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=32768`. Do not mix
those flags into the balanced command unless the user picks that recipe.

## Pier YAML

Copy [scripts/pier-job.yaml.template](scripts/pier-job.yaml.template).

- Validation: `n_attempts: 1`, `n_concurrent_trials: 1`, and `task_names` set to one task (default filter `*httpx-streaming-json-iteration*`).
- Full job: remove `task_names`, set the confirmed attempts and concurrency, use a new `job_name`.

`model_name` stays `hosted_vllm/<served-model-name>` for both engines. That
prefix is the OpenAI-compatible client, not the server. `api_base` is the
relay URL plus `/v1`. Keep `extra_python_packages: [mcp]` when the agent is
mini-swe-agent 2.4.5; that import is not in the task images.

## vLLM source build failures

Stop after one retry. Delete only the container this run created, then rerun.
This section does not apply to SGLang.

- `dubious ownership` on `/opt/vllm`: the script adds `safe.directory`. If it still fails, the mount is wrong.
- `elfutils/libdwfl.h` missing: set `INSTALL_BUILD_DEPS=1`.
- `apt-get update` cannot reach `http://ports.ubuntu.com`: the script rewrites `http://ports.ubuntu.com` to `https://ports.ubuntu.com` in `/etc/apt/sources.list.d/ubuntu.sources` once, then updates again.
- Do not `pip install` a new torch over the image.
