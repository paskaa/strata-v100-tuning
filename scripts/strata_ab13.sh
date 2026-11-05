#!/bin/bash
# 13 开关的交替 A/B：A = config 自带的 6 个（不含官方表外 9 个），B = + 9 个（官方 #3411 表全 13 个）
# 每臂：换 env 文件 → 重启（strata_run.sh）→ 等 READY → 校验 /proc env → bench2 → 记录
# 用法: bash /root/strata_ab13.sh 2      # 参数 = 轮数（每轮 A 一次 + B 一次）
OUT=/tmp/strata_ab13.out
: > $OUT
CFG=/root/Strata-upstream/strata-unsloth-ud-q4_k_xl.json
ROUNDS=${1:-2}
say() { echo "[$(date +%T)] $*" | tee -a $OUT; }

arm_env() {  # $1 = A|B
  if [ "$1" = "B" ]; then cp /root/strata_switchesB.env /root/strata_switches.env
  else : > /root/strata_switches.env; fi
}

verify_arm() {  # $1 = A|B  -> 打印进程里实际有几个表外开关（应为 0 或 9）
  local P want=0 got
  P=$(pgrep -f 'engine-cuda12/strata --serve' | head -1)
  [ -z "$P" ] && { echo 0; return; }
  got=$(tr '\0' '\n' < /proc/$P/environ | grep -cE '^STRATA_(SIDE_MIXER|GR_UP75|B6_MMVQ_ROWS|GU_SWIGLU|QSA_ROWS|ATTN_DAILY|QIL|DEQUANT_WIDE|GR_DOWN_TG)=')
  [ "$1" = "B" ] && want=9
  [ "$got" = "$want" ] && echo ok || echo "MISMATCH(got=$got want=$want)"
}

run_arm() {
  local arm=$1 rep=$2 res
  say "############ ARM $arm rep$rep"
  arm_env $arm
  bash /root/strata_run.sh "$CFG" --no-install >/tmp/ab_${arm}_${rep}.log 2>&1
  res=$(verify_arm $arm)
  say "env-check: $res"
  if [ "${res#ok}" != "" ]; then say "ARM $arm rep$rep INVALID (env)"; return; fi
  say "engines: $(pgrep -fc 'engine-cuda12/strata --serve')  gateway: $(pgrep -fc 'serve/server[.]py')"
  /usr/bin/python3 -u /root/strata_bench2.py >> $OUT 2>&1
  say "---- ARM $arm rep$rep END"
}

for r in $(seq 1 $ROUNDS); do
  run_arm A $r
  run_arm B $r
done
say "ALLDONE"
# 收尾回到 B（13 开关 = 我们要的默认）
arm_env B
bash /root/strata_run.sh "$CFG" --no-install >/tmp/ab_final.log 2>&1
say "restored B: $(verify_arm B)"
