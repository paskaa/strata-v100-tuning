#!/bin/sh
# 2026-11-07: 旧模型已迁移到 UD-IQ4_XS-v31（v0.2.11），此脚本转至新启动脚本。需要旧模型见 run-unsloth-ud-q4_k_xl.sh.bak-v211。
# 监控网关的启动命令请改为 run-unsloth-ud-iq4_xs-v31.sh。
echo "[strata] old UD-Q4_K_XL starter retired 2026-11-07, delegating to run-unsloth-ud-iq4_xs-v31.sh" >&2
exec /root/Strata-upstream/run-unsloth-ud-iq4_xs-v31.sh
