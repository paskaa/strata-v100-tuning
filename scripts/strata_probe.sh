#!/bin/bash
# 用法: bash /root/strata_probe.sh <label>
# 装 build-opt/strata -> 用 cfg_final(34GiB, layer_split 20) + B 开关重启 -> 探针 4K -> 判 SAFE/CRASH
LABEL=${1:-test}
OUT=/tmp/probe_$LABEL.out
: > $OUT
say() { echo "[$(date +%T)] $*" | tee -a $OUT; }

cp /root/strata_switchesB.env /root/strata_switches.env
bash /root/strata_run.sh /root/Strata-opt/cfg_final.json >>$OUT 2>&1
code=$(curl -s --noproxy '*' -m 4 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/v1/models)
[ "$code" = "200" ] || { say "NOT READY ($code)"; exit 1; }
say "ready"
PID0=$(pgrep -f 'engine-cuda12/strata --serve' | head -1)
MARK=$(wc -l < /root/Strata-upstream/strata-unsloth-ud-q4_k_xl.log)
/usr/bin/python3 - >>$OUT 2>&1 <<'PY'
import json,uuid,urllib.request
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
text="[%s] "%uuid.uuid4().hex[:8]+("The quick brown fox jumps over the lazy dog. "*400)+"\nSummarize in one sentence."
body=json.dumps({"model":"qwen3.8-flash-next-unsloth-ud-q4_k_xl","messages":[{"role":"user","content":text}],"max_tokens":16}).encode()
try:
    d=json.loads(op.open(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=900).read())
    t=d.get("timings",{})
    print("probe ok: prompt_n=%s prefill=%.1f decode=%.1f" % (t.get("prompt_n"),t.get("prompt_per_second",0),t.get("predicted_per_second",0)))
except Exception as e:
    print("probe FAILED:", e)
PY
sleep 6
PID1=$(pgrep -f 'engine-cuda12/strata --serve' | head -1)
errs=$(tail -n +$((MARK+1)) /root/Strata-upstream/strata-unsloth-ud-q4_k_xl.log | grep -c "cuBLAS status")
say "pid before=$PID0 after=$PID1 cuBLAS=$errs"
if [ -n "$PID1" ] && [ "$PID0" = "$PID1" ] && [ "$errs" = "0" ]; then say "VERDICT[$LABEL]: SAFE"; else say "VERDICT[$LABEL]: CRASH"; fi
