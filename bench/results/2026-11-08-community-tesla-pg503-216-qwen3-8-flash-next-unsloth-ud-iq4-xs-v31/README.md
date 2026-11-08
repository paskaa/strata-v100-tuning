# Community benchmark on Tesla PG503-216

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: Tesla PG503-216, 32 GiB VRAM
- CPU: Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz (44 cores, 88 threads); installed RAM: 125.6 GiB; storage: sda  PERC H330 Mini
- OS: Linux 6.8.0-147-generic; GPU driver: 580.196.07; CUDA (driver supports up to): 13.0
- Strata: project 0.1.57, commit c561b9c2; engine 0.1.58
- GPU detail (after the run, near idle): power_limit_w=220.00, power_draw_w=58.37, pcie=gen3 x8, pstate=P0, temp_c=60
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-iq4_xs-v31
- Context: 131072 tokens; parallel slots: 2
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
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 66.7 (47.6-99.4) | 0.50 (0.47-0.95) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 971.5 (920.0-976.9) | not measured | 4.17 (4.15-4.40) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1701.9 (1618.7-1704.8) | not measured | 9.49 (9.47-10.22) |
| agent session, follow-up turns | 13123 (13089-13140) at the last turn | see below | up to 64 | 3 | see below | 68.8 (52.6-84.9) | 3.99 (3.53-4.71) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.9 % (68.9-69.1); earlier-turn tokens read again per session: 144 (73-144) (earlier turns that the cache did not hold; the 12276 (12276-12277) new tokens of the files are read once); first turn (nothing cached) first token: 3.11 s (2.75-3.15).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 223.9 | 2.748 | 43.6 |
| 1 | 3379 | 624 | 2755 | 698.9 | 3.987 | 52.6 |
| 2 | 4912 | 3372 | 1540 | 427.2 | 3.643 | 61.1 |
| 3 | 8156 | 4976 | 3180 | 747.3 | 4.295 | 58.8 |
| 4 | 10363 | 8199 | 2164 | 597.6 | 3.66 | 62.2 |
| 5 | 13123 | 10413 | 2710 | 678.4 | 4.032 | 71.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 162 s.
