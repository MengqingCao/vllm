# MiniMax-M3.1-Flash 1P1D (TP4+TP4) + Mooncake Store：相对 SA 镜像 vLLM 的改动

## 基线

- 镜像：`vllm/vllm-openai-rocm:minimax-m31-flash@sha256:ae8681212aa5291d8c27f06e90016435e3fca144a68fd5c439231f6475f74569`
  （来自 InferenceX run 39929767608）
- 镜像内 vLLM：`0.32.0rc1.dev364+g8d0f05a0c.d20261023`。本分支基于 `8d0f05a0c`；从镜像中拷出的
  26 个 `kv_connector/v1`（含 `mooncake/`、`mooncake/store/`）源文件与该 commit 逐字一致。
- 镜像内 Mooncake：`mooncake-transfer-engine-rocm 0.3.13.post1`（未开启 `ENABLE_MULTI_PROTOCOL`）。
- 模型：`amd/MiniMax-M3.1-Flash-Quark-MXFP4` @ `1442d09d4926cb286fa2dc6646c83c4af2b3898c`。
- 配置：主 KV NVFP4、indexer NVFP4、DSpark draft FP8（TRITON_ATTN）、主 attention ROCm NVFP4；
  P 为 MultiConnector（MooncakeConnector `kv_producer` + MooncakeStoreConnector `kv_both`），
  D 只用 MooncakeConnector `kv_consumer`；每个 P rank 的 Store segment 224,873,414,656 B。

实际跑测时**镜像内的包一律未改**。所有改动都以运行时方式生效：通过
`--worker-cls pd_vllm_worker.StoreMemoryWorker` 的 worker 子类，以及 ATOMesh 的启动脚本。
`atomesh/`、`config/`、`cells/` 是跑通时所用文件的原样拷贝（ATOM 本地分支
`mengqing/m31-vllm-pd-mooncake-20261027` @ `f49b47b59`）。本分支的第一个 commit 把其中两处 vLLM
行为改动移植进了 vLLM 源码，**这部分移植本身没有上机运行过**。

## 改动清单

### 1. Store 对象只放在本 rank 自己的 segment 上（已移植：`prefer_local_segment`）

- 现象：job 11048 中，Store PUT 大量失败，报 `batch_put failed ... codes={-800}`（`TRANSFER_FAIL`），
  同时 `hip_transport.cpp: Requested address ... not found!`。
- 原因：Mooncake 0.3.13.post1 没有开启 `ENABLE_MULTI_PROTOCOL`，`HipTransport::install` 会把本进程
  segment 的 protocol 直接改写为 `"hip"`。目标是同机其他 P 进程的 host segment 时，传输会走 HIP，
  而 HIP 只登记了 device buffer，所以找不到地址。
- 改法：Store 初始化后，把 `ReplicateConfig.preferred_segment` 设为 `store.get_hostname()`。
  key 里带 `tp_rank`，每个 rank 只读写自己的分片，因此总容量和单 rank 容量都不变。
- 运行时实现：`atomesh/pd_vllm_worker.py::prefer_own_store_segment`。

### 2. Store 保存前做一次设备同步（已移植：`save_device_sync`）

- 现象：job 11054 中，GPU reset 之后从 Store 重载得到的输出是 `' Answer: Answer: ...'` 这种乱码；
  而同一批块经 P→D 传到 D 时结果是对的。两边每块的字节数一致（1,822,720 B）。
- 改法：`wait_for_save` 在 record 保存事件之前先执行 `torch.cuda.synchronize()`。加上之后
  （job 11075、11156），Store 重载的输出恢复正常。
- 代价：有保存的 step 会失去计算与 I/O 的重叠，C32 的性能数据因此可能偏保守。
- 遗留：还没有定位到具体是哪个 stream 上的 KV 写入没被事件覆盖（V2 model runner + DSpark +
  ROCm NVFP4）。
- 运行时实现：`atomesh/pd_vllm_worker.py::fence_store_saves`。

启用移植版的方法：在 `config/prefill-kv.json` 里 MooncakeStoreConnector 的
`kv_connector_extra_config` 中加入：

```json
"prefer_local_segment": "true",
"save_device_sync": "true"
```

### 3. KFD pin adapter（未移植，属于 Mooncake/amdgpu 层的 workaround）

- 现象：job 10929 中，D 注册约 193 GiB 的 KV 之后，KFD 报告的空闲显存被清零，随后 Triton 加载
  kernel 时报 HIP 209。
- 改法：`ATOMESH_VLLM_KFD_PIN_ADAPTER=1`，包住 `ensure_kv_transfer_initialized`，在外层 KV connector
  上按每块分配建一个 peer-memory pin（机制与 Mooncake PR #4657 相同），并在注册后核对 KFD 空闲显存。
- 这与 README 原先写的"默认不启用"不同。
- 运行时实现：`atomesh/pd_vllm_kfd_pin.py`，在 `pd_vllm_worker.py` 中绑定。

### 4. Store 注册前的 host 内存准备与 NUMA 放置（未移植，属于 ATOMesh 的主机环境处理）

- 现象：job 10945 中，P 的 Store segment 没有拿到大页，ionic 网卡注册到第 167 个 1 GiB chunk 时
  报 `Invalid argument`，随后 `Failed to mount segment: INVALID_PARAMS`。
- 改法：在 `StoreMemoryWorker.initialize_from_config` 中，8 个 P/D worker 加载完模型后一起等待，
  选出一个 worker 统一准备：
  - 先对模型快照和 `/share_nfs/models` 做 fadvise 清理 page cache；
  - 每个 NUMA node 按 1/N 做 THP 预取，单个 node 限时 600 s；
  - 有 node 失败时，在成功的 node 上按整个 Store 池补一轮（1800 s），把它们记为 `store_nodes`，
    P 初始化 Store 期间对这些 node 使用 `MPOL_BIND`；
  - 没有可用 node 时直接拒绝挂载 Store。
- Spur 的容器是非特权的（uid 300165、`CapEff=0`、`/proc/sys` 只读、主机 sudo 需要密码），
  所以 `drop_caches` 和 compact 做不了，只记录为 `unavailable`。
- 运行时实现：`atomesh/pd_vllm_memory.py`、`numa_exec.py`、`compact_huge_pages.py`、
  `drop_page_cache.py`；结果写入 `memory-preparation.json` 和 `*-store-numa.json`。

### 5. 环境变量与配置（与 SA / CI 参考的差异）

- 去掉了 `MC_IB_PCI_RELAXED_ORDERING`：它是在定位 job 10945 时去掉的，是否必要没有单独验证过。
- `MC_MAX_MR_SIZE`：P 为 1 GiB，D 为 256 MiB；`GLIBC_TUNABLES=glibc.malloc.hugetlb=1` 只设给 P；
  每个 worker 设置 `MC_TE_FILTERS=rdma<物理 GPU>`；Store 的 `device_name` 为空（与 CI 40376309047
  一致）。
- smoke 阶段设置 `VLLM_SERVER_DEV_MODE=1`：`/reset_prefix_cache` 属于 dev router，不开会返回 404。
- 启动 AgentX 时去掉值为空的 `AIPERF_*` 变量，否则 AIPerf 会报
  `CONFIGURATION_TIMEOUT float_parsing`（job 11156 的 C32 阶段）。

### 6. smoke 判定（按 owner 决定修改）

PD、独立 D 计算、GPU reset 后的 Store 重载，三条路径的输出开头都必须答对（`4` 或 `four`），
且 Store GET 字节数大于 0。逐字完全一致只作记录：在 `ignore_eos` 下，答完之后的 token 几乎打平，
P 与 D 的 prefill 数值差异会导致分歧（job 11054、11075）。

## 结果（MI355X，单节点 8 卡）

- smoke（job 11156，pit2-p03-g34）：1K、32K、128K 三个长度下三条路径都答对；GPU reset 后 Store GET
  分别为 43.7 MB、1.85 GB、7.45 GB；P/D 都选中 ROCm NVFP4；每个 worker 只发现 1 张 HCA。
- C32（job 11167，pit2-p03-g51，3600 s，393 条 AgentX trace，seed 42，数据集 revision `23f152f6`）：
  - 4522 个请求，0 errors / 0 cancelled，1.25 req/s；
  - 输入 216,725 tokens/s，输出 1,280 tokens/s，按 8 卡计每 GPU 总吞吐 27,251 tokens/s；
  - TTFT 均值 999 ms / p50 769 ms / p99 5,042 ms；ITL 均值 5.63 ms / p99 7.51 ms；
  - GPU 前缀缓存命中约 97%，Store 外部缓存命中约 51%；
  - 这一轮 g51 的 NUMA 1 准备超时，Store 绑定到了 NUMA 0（838 GiB 预取用了 39 s）。

## 运行环境方面的发现

- 有些节点上跑着长期运行、不归 Slurm 管的容器（g12 的 `muse-*` 占满 page cache；g42 的
  `yixingx_fly2` 占 93% 显存）。g40 在加载权重时容器两次被 kill（137）。这几台都已排除。
- 在 Spur 上，作业偶尔会出现 `JobLaunchFailure (gpu/resource allocation mismatch)`，requeue 5 次后
  进入 hold，需要手动 `scontrol release`。
