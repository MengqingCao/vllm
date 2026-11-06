---
name: deepswe-vllm-eval
description: >-
  Run a two-host DeepSWE accuracy eval. The GPU host serves with vLLM or
  SGLang, chosen from the user's request or by asking before launch.
  Pier and mini-swe-agent run on a separate x86 machine. Use when the user
  asks to deploy or continue a DeepSWE accuracy run, serve vLLM or SGLang
  for DeepSWE, reproduce a MiniMax-M3.1-Flash DeepSWE eval, or names
  a GB300 tray and a task host such as hjbog. Asks for host, SSH key path,
  jump host, and local log path, and confirms engine, image, serve config,
  and DeepSWE version before starting the server.
---

# Two-host DeepSWE accuracy eval

The chosen engine serves on the GB300. DeepSWE task images are amd64, so Pier
runs on a separate x86 machine and reaches the server through an SSH tunnel
and a jump host.

## Requirements

使用用户原话，缺一项就先问，问齐之前不要 SSH、不要拉镜像、不要起服务：

1. 让用户提供自己GB300的机器ip、user name、ssh公钥地址（不要接受明文）、jumper ip
2. 让用户提供 deepswe 的测试请求发送机器 ip、user name、ssh公钥地址，对应的是我们用的 hjbog 机器，如果连通性或者有任何功能性阻塞，要通知用户改用其他机器或降低并发数等
3. 让用户提供日志的本地存放路径，或者使用你默认构造的项目目录下的路径
4. 起 vllm 服务之前，给用户机会指定具体配置、具体镜像、deepswe版本等信息

不要从旧对话推断主机、密钥或日志目录。用户没指定日志路径时，用当前项目下的
`experiments/deepswe/<YYYYMMDD>/`。

jumper 的 SSH user 默认与 GB300 user 相同，端口默认 22。只在用户说明时改。
用户明确说没有 jumper 时，才直连 GB300。

### ssh公钥地址

只要本机文件路径，传给 `ssh -i`。不要接受明文：对话里出现 `-----BEGIN`，或一行以
`ssh-ed25519`、`ssh-rsa`、`ecdsa-sha2-nistp256`、`sk-ssh-ed25519` 开头，就拒绝，
请用户改发路径。不要读取密钥文件，不要把内容写进回复、脚本、日志或远程文件。
路径以 `.pub` 结尾时，请用户给对应的私钥文件路径，仍然不要粘贴内容。

操作机到 GB300、操作机到请求发送机器，各自用用户给的路径，加上
`BatchMode=yes` 和 `IdentitiesOnly=yes`。不要把私钥拷到任何服务器。
请求发送机器上的隧道使用那台机器上已经可用的 SSH 凭据。那边认证失败就停，
通知用户，不要代为复制密钥。

## Engine

在启动之前给用户选项，或根据用户的要求自己去选：起 vLLM，还是 SGLang。
后面对应的参数按照对应的引擎去设置。

用户这次已经点名引擎时，用那个引擎，不要再问一遍。没点名时，先只问这两个选项，
选定后再列参数。只使用 [reference.md](reference.md) 里该引擎的 flag 和安装方式。
不要把 vLLM 的 `--tensor-parallel-size`、`--kv-cache-dtype nvfp4` 或
`--speculative-config` 套到 SGLang 上，也不要把 SGLang 的 `--tp`、
`--speculative-algorithm` 套到 vLLM 上。

## Confirm before serving

选定引擎之后、拉镜像和启动服务之前，把将要使用的配置列给用户，等用户明确同意
或改完。MiniMax-M3.1-Flash DeepSWE v1.1 的分引擎默认值在
[reference.md](reference.md)。至少包括：

- 引擎（vLLM 或 SGLang）
- 基础镜像；该引擎的代码来源（git URL 和 ref，或镜像里已经装好的版本）
- 模型 id、revision、HF cache 在 GB300 上的路径
- 该引擎自己的 serve 参数（并行、上下文长度、KV cache、speculative decoding、监听地址）
- DeepSWE git URL 和 commit、Pier 版本、agent 版本和额外 Python 包
- `n_attempts`、`n_concurrent_trials`、采样（temperature、top_p、reasoning）
- jumper user 和端口

先前的“可以开始”不算这张清单的确认。

## Request path

```text
task container (no-network)
  -> http://<docker-bridge>:443/v1    socat on the task host
  -> 127.0.0.1:18000                  ssh -L on the task host
  -> jumper
  -> GB300 127.0.0.1:8000             chosen engine
```

`api_base` 的目标端口只能是 80 或 443。Pier 的出口代理拒绝其他端口
（`ERR_ACCESS_DENIED`），所有 trial 会失败。优先
`http://172.17.0.1:443/v1`。该地址已被占用时，在 `lo` 上添加
`172.31.0.1/32`，仍转发 443。两处都不是 HTTP 200 就停，通知用户。不要改用其他端口。

从操作机经 jumper 登录 GB300：

```text
ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=20 -i <key-path> -o "ProxyCommand=ssh -o BatchMode=yes -o IdentitiesOnly=yes -i <key-path> -p <jumper-port> -W %h:%p <jumper-user>@<jumper-ip>" <gb300-user>@<gb300-host>
```

脚本用 `scp -i <key-path>` 上传。不要把脚本嵌进 PowerShell 的 ssh 参数里。

## 1. Engine on the GB300

确认 HF cache 路径在确认清单里，且本机用户可读。12 小时租约不包含权重。

上传 [scripts/serve.sh](scripts/serve.sh)，按确认结果导出环境变量后执行。
脚本会拉镜像。只有引擎是 vLLM 且确认要从源码安装时，才设置 `INSTALL_VLLM=1`，
把 vLLM 装进镜像自带的 PyTorch。SGLang 保持 `INSTALL_VLLM=0`。
然后跑 `/logs/run-serve.sh`。容器名已存在则退出。不要停其他用户的容器。

`run-serve.sh` 从对应模板生成：

- vLLM：[scripts/run-serve-vllm.sh.template](scripts/run-serve-vllm.sh.template)
- SGLang：[scripts/run-serve-sglang.sh.template](scripts/run-serve-sglang.sh.template)

GB300 上就绪检查：`curl -s http://127.0.0.1:8000/v1/models` 返回确认的模型。
返回之前不要把隧道指过来。vLLM 源码编译失败时看 [reference.md](reference.md)。
用户没确认的话，不要换掉镜像里的 PyTorch，不要设 `VLLM_USE_PRECOMPILED`。

## 2. Tasks on the request machine

`uname -m` 必须是 `x86_64`。不要在 GB300 上跑 DeepSWE 容器，不要为了分数去装
qemu-user binfmt，也不要把任务镜像重做成 arm64。

在请求发送机器上的可写目录（默认 `~/deepswe-eval/{dataset,jobs,logs}`）：

1. 检出用户确认的 DeepSWE commit，核对接 `task.toml` 数量。
2. 后台跑 [scripts/tunnel.sh](scripts/tunnel.sh)。它断线会自己重连。
3. `relay` 容器不存在时跑 [scripts/relay.sh](scripts/relay.sh)。
4. 安装确认过的 Pier 版本。

Pier 作业前，在请求发送机器上：

```bash
curl -s -m 15 http://127.0.0.1:18000/v1/models
curl -s -m 15 -o /dev/null -w '%{http_code}\n' http://<relay-addr>:443/v1/models
```

第二条必须是 `200`。用 [scripts/pier-job.yaml.template](scripts/pier-job.yaml.template)
生成配置。先跑 1 个任务、1 次尝试。该 trial 出分后，再把 `job_name` 换成新名字
跑完整作业。同一份服务上不要并行第二份 `pier run`。

`df` 看 docker 数据盘。并发镜像会占大量磁盘，即使 `environment.delete=true`。

## When something blocks

停下来通知用户。用户表态之前，不要换机器，不要自己降低
`n_concurrent_trials`，不要把作业放到 jumper 上。

下面任一情况，告诉用户改用其他机器或降低并发数，并写明是哪一项检查失败：

- 到 GB300、jumper 或请求发送机器的 SSH 失败或不稳定
- 请求发送机器不是 x86_64，或 Docker 跑不了任务镜像
- 出口代理在 80 和 443 上都到不了推理服务
- 磁盘、内存或 CPU 撑不住确认的并发
- 数据集、Pier 或隧道在这台机器上做不起来

共享的请求发送机器上，不要停其他用户的容器。

## 3. Copy results back

实验结束或确定停掉之后，把日志和精度结果 `scp` 到 intake 里的本地目录：

- GB300：这次的 log 目录（服务日志、安装日志、引擎 SHA 或版本）
- 请求发送机器：Pier 的 `jobs/` 和 `logs/`

沿用上面的 `ssh -i` 和 ProxyCommand。不要拷贝密钥文件。拷完先列出本地目录，
再报告结果。

通过只认 `verifier_result.rewards.reward == 1`（F2P 和 P2P 测试全部通过）。
`partial` 是通过比例，不是通过。按每题 `started_at` 区分第 1 次和第 2 次尝试，
两次分开报，再报合并通过率。

## Constraints

- 推理服务只绑 `127.0.0.1:8000`。
- 不要在 GB300 上跑 DeepSWE 容器。
- 不要在 jumper 上跑 Pier 作业。
- 用户没确认时，不要开 synthetic acceptance。
- 不要输出或保存 SSH 密钥内容。
