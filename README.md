# Strata on 2× Tesla V100 (sm_70) — 调优与上游 PR 落地报告

> 机器：Dell R730 / Ubuntu 24.04 / 2× Xeon E5-2696 v4（88 线程 / 2 NUMA / 无 AVX-512）
> GPU：2× Tesla V100-32GB（PG503-216），**PCIe 非对称：GPU0 Gen3 x8 / GPU1 Gen3 x16**
> 模型：Qwen3.8-Flash-Next UD-Q4_K_XL（GGUF，type 7/8/12/13，非 IQ）
> 引擎：Strata 0.2.10（build 0.1.56），`engine-cuda12/strata`，CUDA 12.8 / sm_70
> 日期：2026-11-05 ~ 11-06

---

## 1. 一句话结论

把上游 14 个 open PR 逐条评估后，**9 个编进本地内核并换装上线**；期间定位并排掉了两个真问题
（一个 GPU MMU fault 由 PR #4522 引起，一个常驻预算 50 GiB 的隐性回归），最终线上稳定在
**layer split 20 + DFlash2（spec 4 / conf 0.5）+ 34 GiB 常驻 + 13 个 sm_70 开关**：
4K prompt prefill **≈900 t/s**、decode **≈62~80 t/s**、draft 接受率 55~70%。

需要先说清的一点：**上游那套 sm_70 开关表（#3411）在我们这套配置上没有可测出的收益**——
A/B 各两轮，prefill 四档差 < 0.6%，decode 均值 68.7 vs 70.8（噪声 ±15%）。上游 +10.7% 的数是在
双 V100-PCIe + **IQ3_S + MTP4 + 单 lane** 上测的，与我们的 Q4_K_XL + DFlash2 不是一回事。

---

## 2. 上游 PR 处理结果

| PR | 内容 | 处理 | 理由 |
|---|---|---|---|
| #5119 | all-cards split engine 支持 sm_70（opt-in） | 编入 | 已编出 `strata-split-2-sm70`；见 §6 为何没启用 |
| #5158 | split engine 每池保留一个窗口的专家 | 编入 | #5119 的依赖 |
| #5258 | 混合量化分池下限 `FNX_POOL_MIN_RESIDENT` | 跳过 | 只对 IQ3_S 这类混合量化加载失败有用 |
| #4522 | 分层下每卡只留本层 dense 权重 + 裁剪后重载 | **回退** | **实测元凶**：GPU0 首次 prefill 必挂 Xid 31 |
| #4549 | 末卡为 draft head 预留显存 | 编入 | 我们 DFlash2 drafter 就在末卡 |
| #2427 / #3559 | `HC_Q8=1` 在分层下不再白留 bf16 副本、只上传本层 Q8_0 行 | 编入 | 省每卡 ~0.6 GiB 显存 |
| #4844 | 自适应轮在流水线 decode 结束后丢专家的修复 | 编入 | 常驻模式的长期稳定性 |
| #3648 | `STRATA_PF_PEER_PIECES=1`（无 P2P 时 prefill 分 8 片） | 回退 | 只在 `--peer-device` 模式生效，我们不走那条路 |
| #3185 | `STRATA_ARENA_RELEASE_BLOCK` 交还整层驻留块 | 回退 | 省的是我们富余的 RAM；与崩溃排查同批，价值低 |
| #2242 | `--numa-hot-node N` 常驻补集按 GPU 所在 NUMA 预占 | 编入 | 附带修掉预算钳制的重复计算 |
| #4258 | `--drop-file-cache` | 编入但**不启用** | 会把 prefill 依赖的 page cache 交还内核，方向相反 |
| #4928 | `--lookup-max-t` 长窗口跨 layer split | 编入 | 配套的 #4040 策略补丁与本地 draft 分支冲突，未并 |
| #5118 | DFlash2 在 layer split 上可与多请求并存（纯 Python） | 回退 | 应用层据此重写配置、把 drafter 换成了 MTP |
| #4105 | `--ple-io ram` 用 O_DIRECT 冷读 PLE 表 | 未并 | 与本地已有的 mlock/rlimit 改动冲突；只影响冷启动 |

---

## 3. 最终配置（线上生效）

```jsonc
// /root/Strata-upstream/strata-unsloth-ud-q4_k_xl.json（应用会重写 env，故开关放启动脚本）
{
  "port": 8080,
  "layer_split": "20",
  "user_args": ["--spec", "4", "--draft-conf", "0.5"],
  "args": ["--pack ...", "--expert-cache auto", "--prefill auto", "--max-context 262144",
           "--kv int8", "--ple-io ram", "--kv-resident 32768",
           "--resident-budget-gib 34",          // ← 关键：不要写 50（见 §5.2）
           "--vram-reserve-mib 1536", "--prompt-cache 16", "--conversation-cache-mib 16384",
           "--vision"]
}
```

`/root/strata_switches.env`（由 `run-unsloth-ud-q4_k_xl.sh` source，保证每次重启都进进程）：

```sh
export STRATA_SIDE_MIXER=2
export STRATA_GR_UP75=1
export STRATA_B6_MMVQ_ROWS=1
export STRATA_GU_SWIGLU=1
export STRATA_QSA_ROWS=1
export STRATA_ATTN_DAILY=1
export STRATA_QIL=1
export STRATA_DEQUANT_WIDE=1
export STRATA_GR_DOWN_TG=1
```

加上 config 自带的 6 个（`SM70_TABLE` / `GR_NORM_FAST` / `ARGMAX_ROWS_SM70` / `EXPDEC` / `HC_Q8` / `FUSE_DECODE`），
引擎进程里共 19 个 `STRATA_*` 变量（含 2 个继承、1 个 READY 看门狗开关）。

---

## 4. 实测数据

### 4.1 13 开关交替 A/B（4 臂，A/B/A/B，每臂重启后从 /proc 取证）

| prompt | A prefill | B prefill | A decode | B decode |
|---|---|---|---|---|
| 130 | 83.8 / 83.0 | 83.1 / 84.3 | 60.8 / 63.2 | 67.8 / 53.5 |
| 1072 | 280.2 / 280.8 | 279.4 / 280.2 | 64.0 / 73.2 | 79.3 / 75.1 |
| 4071 | 896.7 / 898.1 | 897.5 / 895.2 | 74.0 / 61.8 | 66.0 / 73.0 |
| 8071 | 1235.4 / 1232.3 | 1233.0 / 1226.0 | 80.8 / 71.6 | 79.2 / 72.5 |
| 均值 | 83.4 / 280.5 / 897.4 / 1233.9 | 83.7 / 279.8 / 896.4 / 1229.5 | 68.7 | 70.8 |

- prefill **无差异**（<0.6%，两轮内部重复性 ±0.3%）
- decode 均值 +3%，但同臂两轮可差 12 t/s、不同尺寸符号来回翻 → **噪声内，无可测收益**
- 输出一致性：固定 3 条 prompt（temp 0）与换内核前比对，2/3 逐字相同，第 3 条只差一个 token（语义等价）

### 4.2 探针（恢复后回归）

4,069-token prompt → prefill **922.9 t/s**、引擎 pid 不变、`cuBLAS status` 计数 0 → SAFE。

---

## 5. 两个真事故

### 5.1 GPU0 MMU fault（Xid 31 / cuBLAS 13）

```
NVRM: Xid (PCI:0000:04:00): 31 ... MMU Fault: GPC5 GPCCLIENT_T1_0
        faulted @ 0x7da1_ea000000. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_WRITE
prefill gemm: cublasGemmEx f16: cuBLAS status 13
```

- 现象：新内核加载完成后的**第一次 prefill** 就挂，引擎 exit 1，网关自动重启（循环）。
- bisect：从回退基线只回放 **#4522** → 原样复现（pid 1019734，地址 0x7656_a2000000）。
- 结论：`#4522` 在 CUDA0 上释放并重建权重 arena 后，prefill 仍在往旧地址写。该 PR 已回退。

### 5.2 `--resident-budget-gib 50` 的隐性回归

`#2242` 修掉"预算钳制重复计算 arena"之后，写 50 就**真的 pin 50 GiB**（旧内核只 pin 34.10）。后果：

```
expert tiers: ... files 1219 blobs 225024.4 MB read   # 225 GB 读盘
decode 51 tok/s, 1K prefill 117.7 t/s
```

NUMA 预占为了腾地方把模型文件的 page cache 丢了，剩下 21.7 GiB 专家只能从 PERC 读。
改回 **34 GiB**：`resident RAM: 31.81 GiB ... 0 blob reads from the file`，1K prefill 280.8。

### 5.3 应用会重写 config 的 env

`serve/server.py` 启动时把 `env` 改回它自己的 6 个默认值，并可能把 drafter 从 DFlash2 换成 MTP。
→ 新开关必须放 `run-unsloth-ud-q4_k_xl.sh`（网关环境），不能只写 config。

---

## 6. split engine（#5119 + #5158）：编好、装上、被规则挡住

- 产物：`engine-cuda12/strata-split-2-sm70`（23.7 MB），启动需无限 memlock（`prlimit --memlock=unlimited:unlimited`）。
- 配 `"split_engine": "on"` 后应用仍起层切分，决策函数给的原因：

  ```
  Using the normal layer split: this run config has engine arguments of its own or replicas,
  which the all-cards split engine does not take.
  ```

  （`strata_control/split_engine.py:489` 因 `user_args`/`replicas` 拒绝；485 行还拒 `drafting: dflash2`。）
- 结论：这是**另一条路线**，得用 `strata launch --split-engine on` + 精简配置（无 `user_args`、MTP drafter、
  26 GiB host cache、无限 memlock），代价是丢 vision / prompt-cache / conversation-cache。
  上游同款双 V100-32G：prefill 2305 t/s、decode 69.0；服务端短回复 132~133 t/s（层切分 987~996 / 49~66）。

---

## 7. 盘：SSD 但被控制器卡在 106 MB/s

- 物理盘：3× **Great Wall S300 1TB（SSD，9902 小时，SMART 全 PASSED）**，RAID5 → `sda` 2.05 TB（PERC H330 Mini）。
- 内核报 `Write cache: disabled, read cache: disabled`（无 BBU），`rotational=1` 是控制器的假信号。
- 实测：单流 O_DIRECT 1 GiB **106 MB/s**；三路并行各 53 MB/s，**合计仍是 106 MB/s**（硬顶）。
- 推论：引擎那句 `drive 0.15 GB/s` 没估错；34 GiB 常驻补集的冷读下限 ~5.5 分钟，换盘无用。

---

## 8. 复现 / 运维

```bash
# 构建（合并后的树：/root/Strata-opt，分支 v2-safe）
cmake -S . -B build-opt -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_EXPERIMENTAL_SM60=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
ninja -C build-opt -j 40 strata

# 换装 + 重启（停网关/引擎/vision → 装二进制 → 写配置 → 等 models 200）
bash /root/strata_run.sh /root/Strata-opt/cfg_final.json

# 探针（4K prompt，判定 SAFE/CRASH）
bash /root/strata_probe.sh mylabel

# 13 开关 A/B（2 组交替，约 1 小时）
bash /root/strata_ab13.sh 2
```

回滚件：`engine-cuda12.bak-20261105-185753/`、`serve.bak-*/`、`strata_control.bak-*/`、`strata_cli.bak-*/`、
配置 `.bak-r6`。

---

## 9. 注意（重要）

- 每次重启要重读 34 GiB 补集（~5.5 分钟）+ 填显存缓存（~2 分钟），期间 8080 不可用；
  跑调优脚本前先确认没人在用。

---

## 10. 未做

1. split engine 实跑（需单独时段 + 精简配置）。
2. `#3185` / `#3648` 单独回放验证（价值低）。
3. ~~`--numa-hot-node` 的 A/B~~ —— **2026-11-06 已做，无收益，结论：不用**。
   4 臂 × 2 轮交替（A=默认不传参、B=`--numa-hot-node 0`），prefill 全档差 <0.5%（8000-token：1232.0/1232.3/1196.0/1234.4），
   decode 均值 A rep1=68.2、B rep1=68.2、A rep2=71.9、B rep2=68.1 —— 全在 ±15% 噪声内。
   官方 `NUMA placement` 日志显式两节点各分到 ~50%（node0 15.95 GiB / node1 15.86 GiB），已自动均衡，无"单边绑定"问题。
   最终配置保持不带 `--numa-hot-node`，少传一个参数。数据：`/tmp/strata_numa.out`。
4. 用 IQ3_S + MTP4 + 单 lane 复现上游 +10% 的开关收益。
