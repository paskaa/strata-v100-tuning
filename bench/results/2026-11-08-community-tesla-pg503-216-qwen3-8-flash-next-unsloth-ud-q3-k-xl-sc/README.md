# Community benchmark on Tesla PG503-216

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla PG503-216, 32 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz (44 cores, 88 threads); installed RAM: 125.6 GiB; storage: sda  PERC H330 Mini
- OS: Linux 6.8.0-147-generic; GPU driver: 580.196.07; CUDA (driver supports up to): 13.0
- Strata: project 0.1.57, commit c561b9c2; engine 0.1.58
- GPU detail (after the run, near idle): power_limit_w=220.00, power_draw_w=59.38, pcie=gen3 x8, pstate=P0, temp_c=61
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 74.9 (60.6-109.2) | 0.49 (0.47-0.53) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1082.5 (1082.3-1094.7) | not measured | 3.75 (3.70-3.75) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1779.4 (1774.8-1816.7) | not measured | 9.08 (8.89-9.10) |
| agent session, follow-up turns | 13138 (13126-13145) at the last turn | see below | up to 64 | 3 | see below | 68.4 (56.1-104.1) | 3.51 (2.81-4.40) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.1 % (68.7-69.3); earlier-turn tokens read again per session: 74 (4-214) (earlier turns that the cache did not hold; the 12275 (12274-12278) new tokens of the files are read once); first turn (nothing cached) first token: 2.87 s (2.41-2.94).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 257.2 | 2.411 | 45.5 |
| 1 | 3376 | 623 | 2753 | 794.7 | 3.511 | 66.1 |
| 2 | 4909 | 3369 | 1540 | 484.4 | 3.218 | 62.4 |
| 3 | 8153 | 4902 | 3251 | 747.5 | 4.397 | 64.7 |
| 4 | 10381 | 8146 | 2235 | 579.5 | 3.899 | 63.5 |
| 5 | 13138 | 10428 | 2710 | 765.6 | 3.581 | 64.2 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 136 s.
