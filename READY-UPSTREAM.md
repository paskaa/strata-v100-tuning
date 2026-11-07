# 双 V100-32GB + 上游 Strata v0.2.11 上机清单（R730）

> 已升级验证（2026-11-07 实测）：仓库 `/root/Strata-upstream`（分支 `v100-v211` = 上游 v0.2.11 tag + V100 专属移植：sm70 split-engine 显式 opt-in、arena 溢出 host-cache、rope 透传；具名 DFlash2 上游已收编）；
> 引擎本地编译 0.1.57（sm70，CUDA 12.8）；模型已换 **UD-IQ4_XS-v31** 四分片（93.7GB，revision d9620119，size+sha256 全验）
> `/root/Strata-V100/gguf-cache/UD-IQ4_XS-v31/` + pack `unsloth-ud-iq4_xs-v31` + mmproj `vision-cache/`；
> 内存 125GB；磁盘剩 1.4T；电源 2×1100W 在位；功耗墙 220W（写在 `run-unsloth-ud-iq4_xs-v31.sh` 里，重启生效）。
> 上游 v0.2.11 保留 Volta 路径（`STRATA_EXPERIMENTAL_SM60` + CUDA 12 引擎包 + `docs/OLDER_GPUS.md`），
> 且自带 DFlash2（默认推测解码）与 all-cards split 双卡引擎。
> 旧模型 UD-Q4_K_XL 文件保留，旧启动脚本已转接（`run-unsloth-ud-q4_k_xl.sh` → 新脚本，备件 `.bak-v211`）；旧配置 `strata-unsloth-ud-q4_k_xl.json` 留作回退。

## 测速基线（2026-11-07，UD-IQ4_XS-v31，layer-split 20，DFlash2，kv int8，ctx 262K；spec8 三轮中位）

| 档 | decode 中位 | prompt | 备注 |
|---|---|---|---|
| 短聊天 | 86.7 tok/s | — | 接受率 ~62% |
| ~4K | 81.3 tok/s | ~974 tok/s | 接受率 ~66% |
| ~16K | 87.9 tok/s | ~1466 tok/s | 接受率 ~70%，hit 98% |

> spec 8 相对 spec 4 单轮约 +1~2%（噪声范围），无回退、显存余量不变（~1.68GB），予以保留。
> MTP 三轮对比：decode 86.0/88.7/91.9（-1%/+9%/+4.5%），prompt 942/1385（-3%/-5.5%）——速度打平，但 MTP 双卡显存 62GB（余 0.6GB）vs DFlash2 38GB，判 DFlash2 留用。
> 对照：社区 2×V100-16GB UD-Q4（#5367）decode 56~59、prompt 677/1517（4K/16K）；本机 0.2.10+UD-Q4 split20 均值 68.8。结论：升级包出字涨约一成多，读提示持平。

## 卡到后步骤

```bash
# 0. 两张卡：Slot 4（x16）+ Riser 3（x8），各接 8pin；GPU 导风罩 + 高转风扇；iDRAC 关第三方卡风扇联动
# 1. 装 580+ 驱动（v0.2.10 floor 575.51，取 580+；toolkit 保持 12.8不动）
sudo apt-get install -y nvidia-driver-580
nvidia-smi  # 确认两张 V100-32GB，记下 nvidia-smi 编号（按 PCI bus 排）
sudo nvidia-smi -pl 200   # 先给双卡都设 200W 墙（175W 更稳，按温度定）
sudo nvidia-persistenced --user root; sudo systemctl enable nvidia-persistenced

# 2. 装（--gpus 点名或 --cuda 12 即自动走 Volta+CUDA12 路径，不用再 export 开关也行，export 最稳）
cd /root/Strata-upstream
export STRATA_EXPERIMENTAL_SM60=1
export https_proxy=http://127.0.0.1:18080 http_proxy=http://127.0.0.1:18080
python3 setup.py --yes --cuda 12 --gpus 0,1 --model UD-Q4_K_XL --context 262144 --kv int8 \
  --gguf-dir /root/Strata-V100/gguf-cache/UD-Q4_K_XL --no-start
python3 setup.py --calibrate   # 先按默认 layer split 出 baseline
```

## 双卡调优顺序（一次只动一样，记数）

1. 默认 layer split + DFlash2（0.2.10 默认）：记出字/温度/`/metrics` 命中率——这是保底成绩。
2. 再试 all-cards split（0.2.10 新双卡引擎，A6000 上比 layer split 快约一倍，**V100 未验证**）：快就留，慢/报错就退回 layer split。
3. `--kv tq6`（524K 才必要；262K 可跳过）。
4. 出字预期（双 32GB，UD-Q4，calibrate 为准）：80～120 tok/s；读提示 1500～2500。

## 失败预案

- 某张卡认不出：`nvidia-smi -L` 定编号，`--gpus` 改对；再不行单卡 `--gpu 0` 先跑。
- 编译失败：`STRATA_NVCC=/usr/local/cuda-12.8/bin/nvcc` 显式指定；看 `docs/OLDER_GPUS.md`。
- CUDA 13 相关报错一律回 12（sm_70 在 13 已死）；上游 ready-made CUDA13 包与 V100 无关，别装。
