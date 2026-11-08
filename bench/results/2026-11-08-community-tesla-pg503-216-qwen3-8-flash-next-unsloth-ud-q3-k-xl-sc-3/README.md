# Community benchmark on Tesla PG503-216

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla PG503-216, 32 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz (44 cores, 88 threads); installed RAM: 125.6 GiB; storage: sda  PERC H330 Mini
- OS: Linux 6.8.0-147-generic; GPU driver: 580.196.07; CUDA (driver supports up to): 13.0
- Strata: project 0.1.57, commit c561b9c2; engine 0.1.58
- GPU detail (after the run, near idle): power_limit_w=220.00, power_draw_w=59.35, pcie=gen3 x8, pstate=P0, temp_c=63
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q3_k_xl-schemeA
- Context: 262144 tokens; parallel slots: 2
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 72.8 (64.1-115.6) | 0.49 (0.45-0.62) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1090.0 (1074.9-1099.0) | not measured | 3.72 (3.69-3.77) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1773.5 (1773.2-1821.1) | not measured | 9.11 (8.87-9.11) |
| agent session, follow-up turns | 13163 (13137-13167) at the last turn | see below | up to 64 | 3 | see below | 73.9 (66.0-95.3) | 3.44 (2.76-4.36) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1 % (69.0-69.3); earlier-turn tokens read again per session: 74 (3-144) (earlier turns that the cache did not hold; the 12275 (12274-12280) new tokens of the files are read once); first turn (nothing cached) first token: 2.77 s (2.36-2.87).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 261.9 | 2.357 | 46.2 |
| 1 | 3376 | 623 | 2753 | 795.4 | 3.504 | 66.0 |
| 2 | 4909 | 3439 | 1470 | 539.9 | 2.76 | 68.4 |
| 3 | 8153 | 4973 | 3180 | 834.0 | 3.855 | 72.6 |
| 4 | 10374 | 8210 | 2164 | 677.0 | 3.238 | 71.4 |
| 5 | 13137 | 10427 | 2710 | 769.8 | 3.562 | 78.8 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 135 s.
