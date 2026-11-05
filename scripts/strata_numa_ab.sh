#!/bin/bash
# --numa-hot-node 的交替 A/B：A = 不带该参数；B = --numa-hot-node 0
# 每臂：写配置 -> 重启 -> 等 READY -> 从 /proc/<pid>/cmdline 取证 -> bench2
OUT=/tmp/strata_numa.out
: > $OUT
ROUNDS=${1:-2}
say() { echo "[$(date +%T)] $*" | tee -a $OUT; }

run_arm() {
  local arm=$1 rep=$2 cfg flag want got
  say "############ NUMA ARM $arm rep$rep"
  if [ "$arm" = "B" ]; then cfg=/root/Strata-opt/cfg_numa0.json; else cfg=/root/Strata-opt/cfg_final.json; fi
  cp /root/strata_switchesB.env /root/strata_switches.env
  bash /root/strata_run.sh "$cfg" --no-install >/tmp/numa_${arm}_${rep}.log 2>&1
  code=$(curl -s --noproxy '*' -m 4 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models)
  [ "$code" = "200" ] || { say "ARM $arm rep$rep NOT READY ($code)"; return; }
  P=$(pgrep -f 'engine-cuda12/strata --serve' | head -1)
  got=$(tr '\0' ' ' < /proc/$P/cmdline | grep -c -- '--numa-hot-node')
  [ "$arm" = "B" ] && want=1 || want=0
  if [ "$got" != "$want" ]; then say "ARM $arm rep$rep INVALID (numa flag got=$got want=$want)"; return; fi
  say "cmdline-check ok (numa=$got)"
  say "resident line: $(grep -E 'resident RAM mode' /root/Strata-upstream/strata-unsloth-ud-q4_k_xl.log | tail -1 | cut -c1-120)"
  say "numa lines: $(grep -E 'NUMA placement:' /root/Strata-upstream/strata-unsloth-ud-q4_k_xl.log | tail -1 | cut -c1-160)"
  /usr/bin/python3 -u /root/strata_bench2.py >> $OUT 2>&1
  say "---- NUMA ARM $arm rep$rep END"
}

for r in $(seq 1 $ROUNDS); do
  run_arm A $r
  run_arm B $r
done
say "ALLDONE"
