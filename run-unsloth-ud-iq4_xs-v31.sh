#!/bin/sh
cd "/root/Strata-upstream"
# 2026-11-05: 模型在 sda 机械盘上，冷启动读盘 >900s 会被网关的 ENGINE_READY_S 看门狗杀掉
# （"did not report READY within 900s" 后 proc.kill，无任何错误提示）。0 = 无限等。
export STRATA_ENGINE_READY_S=0

# 功耗墙 220W（2026-11-07 起；要改墙只改这里两行，重启服务生效）
# 先起保持进程，否则 -pl 落不住（实测：不开 persistenced 查询还是 300W）
nvidia-persistenced --user root >/dev/null 2>&1 || true
nvidia-smi -i 0 -pl 220 >/dev/null 2>&1
nvidia-smi -i 1 -pl 220 >/dev/null 2>&1

# 开关由 /root/strata_switches.env 决定（A/B 轮次会改它），应用重写 config 也不影响
[ -f /root/strata_switches.env ] && . /root/strata_switches.env


exec "/root/Strata-upstream/.venv/bin/python" "/root/Strata-upstream/serve/server.py" "--engine" "strata" "--config" "/root/Strata-upstream/strata-unsloth-ud-iq4_xs-v31.json" "--port" "8089" "--open"
