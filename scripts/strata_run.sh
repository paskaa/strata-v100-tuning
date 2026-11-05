#!/bin/bash
# 用法: bash /root/strata_run.sh <config.json> [--no-install]
# 停服务 -> (可选)装新二进制 -> 写入配置 -> 拉起 -> 等 READY -> 打印启动摘要
set -u
CFG=${1:?config path}
INSTALL=${2:-install}
DST=/root/Strata-upstream
OUT=/tmp/strata_run.out
: > $OUT
say() { echo "[$(date +%T)] $*" | tee -a $OUT; }

GW=$(pgrep -f 'serve/server[.]py' | tr '\n' ' ')
ENG=$(pgrep -f 'engine-cuda12/strata --serve' | tr '\n' ' ')
VIS=$(pgrep -f 'engine-cuda12/strata-vision' | tr '\n' ' ')
say "kill gateway=[$GW] engine=[$ENG] vision=[$VIS]"
for p in $GW $ENG $VIS; do kill -TERM $p 2>/dev/null; done
sleep 8
for p in $GW $ENG $VIS; do kill -9 $p 2>/dev/null; done
for i in $(seq 1 30); do ss -ltn 2>/dev/null | grep -q ':8080 ' || break; sleep 2; done
say "port listeners: $(ss -ltn | grep -c ':8080 ')"

if [ "$INSTALL" != "--no-install" ]; then
  install -m 755 /root/Strata-opt/build-opt/strata $DST/engine-cuda12/strata || { say "install failed"; exit 1; }
  say "installed engine $(stat -c %s $DST/engine-cuda12/strata) bytes"
fi

cp "$CFG" $DST/strata-unsloth-ud-q4_k_xl.json
say "config written: env=$(python3 -c "import json;print(len(json.load(open('$CFG'))['env']))") split=$(python3 -c "import json;print(json.load(open('$CFG')).get('layer_split'))")"

( cd $DST || exit 1
  export STRATA_EXPERIMENTAL_SM60=1
  export https_proxy=http://127.0.0.1:18080 http_proxy=http://127.0.0.1:18080
  setsid nohup ./run-unsloth-ud-q4_k_xl.sh > /tmp/strata_start_last.log 2>&1 & )
say "launched"
for i in $(seq 1 220); do
  sleep 6
  P=$(pgrep -f 'engine-cuda12/strata --serve' | head -1)
  code=$(curl -s --noproxy '*' -m 4 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models 2>/dev/null)
  if [ -n "$P" ] && [ "$code" = "200" ]; then say "READY pid=$P ~$((i*6))s"; break; fi
  [ $((i % 10)) -eq 0 ] && say "loading pid=${P:-none} models=$code ~$((i*6))s"
done
P=$(pgrep -f 'engine-cuda12/strata --serve' | head -1)
say "cmdline: $(tr '\0' ' ' < /proc/$P/cmdline 2>/dev/null | grep -oE '\-\-(draft|layer-split|spec|resident-budget-gib)[ =][^ ]*' | tr '\n' ' ')"
say "env: $(tr '\0' '\n' < /proc/$P/environ 2>/dev/null | grep -cE '^STRATA_') vars"
say "--- 启动摘要"
grep -iE "layer split:|expert cache [0-9]+ slots|resident RAM mode|VRAM free" $DST/strata-unsloth-ud-q4_k_xl.log | tail -5 | tee -a $OUT
say DONE
