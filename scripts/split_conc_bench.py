#!/usr/bin/env python3
"""Split + Q4_K_XL concurrency bench. POST /v1/chat/completions, temp 0.
Usage: python3 bench/split_conc_bench.py --port 8090 --out bench/results/2026-11-09-split-conc/results.jsonl
Matrix: single x5, conc2 x3 rounds, conc4 x2 rounds. Each request: unique tag (cache-busting),
~200-word prompt, max_tokens 200. Records timings from response (predicted_per_second etc.).
"""
import argparse, json, statistics, time, uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

PROMPT_BODY = (
    "请详细论述双卡大模型推理引擎在长上下文场景下的性能调优方法：先说明 KV 缓存占用随"
    "上下文长度的增长规律，再分析专家权重在多卡之间切分的访存代价，然后给出在显存受限"
    "条件下平衡吞吐与时延的实操建议，最后列出常见故障的定位步骤与对应的缓解措施。"
)

def one(port, model, tag, max_tokens=200):
    prompt = PROMPT_BODY + f" [tag {tag}]"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0,
    }).encode()
    req = Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                  headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
    except Exception as e:
        return {"ok": False, "err": str(e)[:200], "wall_s": time.time() - t0}
    wall = time.time() - t0
    t = data.get("timings") or {}
    usage = data.get("usage") or {}
    return {"ok": True, "wall_s": round(wall, 2),
            "decode_tps": t.get("predicted_per_second"),
            "prompt_tps": t.get("prompt_per_second"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "predicted_n": t.get("predicted_n")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--model", default="qwen3.8-flash-next-ud-q4_k_xl-split-test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 1) if xs else None

    def run_group(name, n_req, rounds):
        print(f"== {name}: {n_req} concurrent, {rounds} rounds ==", flush=True)
        out = {"group": name, "n_req": n_req, "rounds": []}
        for r in range(rounds):
            tag = f"{name}-{r}-{uuid.uuid4().hex[:6]}"
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=n_req) as ex:
                res = list(ex.map(lambda i: one(args.port, args.model, f"{tag}-{i}"), range(n_req)))
            wall = round(time.time() - t0, 2)
            ok = [x for x in res if x.get("ok")]
            rec = {"round": r, "wall_s": wall, "req": res,
                   "median_decode": med([x.get("decode_tps") for x in ok]),
                   "median_prompt": med([x.get("prompt_tps") for x in ok])}
            print(f"  round {r}: wall={wall}s median_decode={rec['median_decode']} n_ok={len(ok)}/{n_req}", flush=True)
            out["rounds"].append(rec)
			# fail fast on engine death
            if not ok:
                print("  !! all requests failed; stopping", flush=True)
                break
        return out

    results = {"port": args.port, "model": args.model, "groups": []}
    results["groups"].append(run_group("single", 1, 5))
    results["groups"].append(run_group("conc2", 2, 3))
    results["groups"].append(run_group("conc4", 4, 2))
    with open(args.out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("WROTE", args.out, flush=True)

if __name__ == "__main__":
    main()
