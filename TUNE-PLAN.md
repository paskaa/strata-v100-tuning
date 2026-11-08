# 双 V100-32GB + Strata v0.2.10 调优执行计划（UD-Q4_K_XL，262K 上下文）

> 原则：一次只动一样；每步先备份配置再改；每步记数（出字/读提示/命中率/温度/功耗）；
> 温度红线 83℃，功耗墙先 200W；单步超时 40 分钟就停，先保可用。
> flag 名以上游 v0.2.10 `setup.py --help` / `docs/DETAILS.md` 为准，本计划写的是 fork 实测过的名字，
> 对不上就按 `--help` 的来，含义不变。

## Phase 0：硬件基线（15 分钟，不装模型先验电）

```bash
nvidia-smi -L; nvidia-smi -q -d PCIE | grep -E "Link (Gen|Width)"
# 期望：两张 V100-32GB；Slot 4 那张 x16 Gen3，Riser 3 那张 x8 Gen3
sudo nvidia-smi -pl 200
sudo nvidia-persistenced --user root; sudo systemctl enable nvidia-persistenced
nvidia-smi dmon -s pucv -d 5 -c 6   # 空载温度/功耗快照，记下来
```

- 门禁：两张都认得出、链宽对（x16/x8）、空载温度 < 55℃。不过就查供电线/槽位/挡片，别往下走。

## Phase 1：单卡冒烟（ isol 坏卡，30 分钟）

```bash
cd /root/Strata-upstream
export STRATA_EXPERIMENTAL_SM60=1
export https_proxy=http://127.0.0.1:18080 http_proxy=http://127.0.0.1:18080
python3 setup.py --yes --cuda 12 --gpu 0 --model UD-Q4_K_XL --context 262144 --kv int8 \
  --gguf-dir /root/Strata-V100/gguf-cache/UD-Q4_K_XL --no-start
```

- 只用 GPU 0 跑通一次短对话即过（出字 Finn... 出字 30+ 即正常，单卡 Q4 半载）。
- 再对 GPU 1 同样来一次（`--gpu 1`）。两张都冒烟通过，进双卡。

## Phase 2：双卡 layer split 基线 + calibrate（1 小时）

```bash
python3 setup.py --yes --cuda 12 --gpus 0,1 --model UD-Q4_K_XL --context 262144 --kv int8 \
  --gguf-dir /root/Strata-V100/gguf-cache/UD-Q4_K_XL --no-start
python3 setup.py --calibrate
```

- 记基准表（短聊天 + 32K 文档，各 3 次取中位）：出字、读提示、`/metrics` 命中率、双卡温度/功耗。
- 预期：出字 70～100，读提示 1000～2000。差太多先查 PCIe 链宽和功耗墙，不调参数。

## Phase 3：推测解码 DFlash2 vs MTP（30 分钟）

- 0.2.10 默认 DFlash2：直接用 Phase 2 的数。
- 切回 MTP 跑同一组（按 `--help` 找开关，如 `--drafter mtp` 类）：差 < 5% 就锁死默认，不折腾。

## Phase 4：spec 4 → 8（20 分钟）

- 64GB 显存够 MTP 开大窗。`--spec 8`（或等价项）跑同一组，出字涨就留，温度/显存报警就退回。
- UD-Q4 在 CUDA12 引擎上确认 `STRATA_MMQ_KQ=1` 生效（setup 自动写，防 #879 非有限 logits；看启动日志有一行即 OK）。

## Phase 5：all-cards split 实测（40 分钟，可跳过）

- 0.2.10 新双卡引擎（A6000 上比 layer split 快约一倍，**V100 未验证**）。
- 按 `--help` 找开关名，用同一组对比 layer split：快 ≥15% 留，持平/慢/报错退回，全程盯温度。
- 这是唯一"实验性"步骤，前面 Phase 都是保底。

## Phase 6：KV（按需）

- 262K + int8 一切正常就跳过。想开 524K 或显存吃紧：`--kv tq6`，省约 1.6GB 给专家缓存，再跑 Phase 2 同一组验证。

## Phase 7：并发（20 分钟）

- `--parallel 2` 起（fork 给 Volta 的默认值），两路同测：单路掉速 < 20% 就留着当日常配置；掉得多就改回 1。

## Phase 8：功耗墙 200 vs 175（15 分钟）

- 同一组在 175W 下重跑：出字掉 < 5% 就钉死 175W（更凉更静）；掉得多就回 200W。

## 记录表（每次填一行）

| 步骤 | 改动 | 出字 | 读提示 | 命中率 | 卡0 ℃/W | 卡1 ℃/W | 结论（留/退） |
|---|---|---|---|---|---|---|---|
| P2 基线 | 默认 | | | | / | / | |
| P3 | | | | | / | / | |
| P4 | | | | | / | / | |
| P5 | | | | | / | / | |
| P7 | | | | | / | / | |
| P8 | | | | | / | / | |

## 熔断

- 任一温度持续 > 83℃：降功耗墙 → 降 spec → 停测查风道，三步走。
- engine 报错含 CUDA/arch：回 Phase 1 单卡定位是哪张卡/哪个槽。
- 总时长超 4 小时还没出基线：停，带着 `/metrics` 和日志找 fork 作者 issue（先搜 V100 相关）。
