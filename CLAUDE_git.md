# vllm-market-agent

## What this repo is for

This repo gets toured live, by its author, under questioning. Optimise every decision for
*defensibility*, not for feature count. A small repo whose every line has a reason
beats a large one that assembles frameworks.

## What the contribution actually is

The contribution is the **serving path**, not the agent.

vLLM is a performance system — continuous batching, PagedAttention, KV-cache
paging — running memory-constrained on Apple silicon (vllm-mlx). The distinctive
output of this repo is *measurement*: throughput vs. batch size, latency
distribution, KV-cache memory accounting, config A vs. config B with the reasoning
for the choice. The sentiment agent is the **harness that generates load**, not the
point.

When there is a choice between making the agent cleverer and making the serving
path better understood, choose the serving path.

## Rules for changes

1. **No new capability** unless it serves the serving-path performance story or the
   engineered slice below. RAG, MCP, LoRA, more tools, more data sources: the
   default answer is **no**. If asked for one, say no and say why *before* writing
   any code. Breadth of buzzwords reads as tutorial-following and is the main risk
   to this repo.
2. **No performance claim without a number.** Every claim needs a reproducible
   benchmark script plus committed results recording hardware, model, quantisation
   and server config.
3. **Every non-obvious decision gets 5–10 lines in `docs/decisions.md`**: what was
   chosen, what was rejected, why. That file is the interview script.
4. **Prefer deleting to adding.** Dead scripts, `.bak` files and tracked
   `__pycache__` are all fair game.
5. Prefer ~40 lines of stdlib to a framework. Frameworks hide the thing being
   measured.

## The engineered slice

Exactly one part of this repo is held to production standard — currently the
**inference client** (the code that talks to the vLLM server):

- typed, documented interface
- explicit handling of timeouts, server errors and malformed model output
- unit tests, including a bad-model-output case
- a benchmark that exercises it

Everything else is explicitly **scaffolding** and is labelled as such in the README.
Do not silently upgrade the scaffolding — ask first. "This part I engineered, this
part is exploratory" is a strong answer in an interview; a uniformly half-polished
repo is not.

