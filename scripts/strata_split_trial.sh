#!/bin/bash
# 启动 split engine 试验：写配置 -> 起 run_split.sh（prlimit 无限 memlock）-> 等 READY 并打印决策日志
OUT=/tmp/split_trial.out
: > $OUT
say() { echo "[$(date +%T)] $*" | tee -a $OUT; }

cp /root/Strata-opt/cfg_split.json /root/Strata-upstream/strata-unsloth-ud-q4_k_xl.json
say "config written: split_engine=$(python3 -c "import json;print(json.load(open('/root/Strata-upstream/strata-unsloth-ud-q4_k_xl.json')).get('split_engine'))")"
cp /root/strata_switchesB.env /root/strata_switches.env
say "switches: $(grep -c '^export' /root/strata_switches.env) exports"

( cd /root/Strata-upstream && setsid nohup /root/run_split.sh > /tmp/split_start.log 2>&1 </dev/null & )
say "launched via run_split.sh (prlimit memlock unlimited)"

for i in $(seq 1 240); do
  sleep 6
  code=$(curl -s --noproxy '*' -m 4 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models 2>/dev/null)
  if [ "$code" = "200" ]; then say "READY ~$((i*6))s"; break; fi
  [ $((i % 20)) -eq 0 ] && say "loading ~$((i*6))s models=$code engine=$(pgrep -fc 'strata-split' 2>/dev/null || echo 0) layer=$(pgrep -fc 'engine-cuda12/strata --serve' 2>/dev/null || echo 0)"
done

say "--- 端口/进程"
say "split procs: $(pgrep -af 'strata-split' | head -3 | cut -c1-120)"
say "layer procs: $(pgrep -af 'engine-cuda12/strata --serve' | head -2 | cut -c1-80)"
say "--- 决策与启动日志"
grep -iE "split engine|split_engine|strata-split|fallback|FNX_|arena|does not fit|host cache" /root/Strata-upstream/strata-unsloth-ud-q4_k_xl.log | tail -12 | tee -a $OUT
grep -iE "split|FNX" /tmp/split_start.log | tail -8 | tee -a $OUT
say DONE
