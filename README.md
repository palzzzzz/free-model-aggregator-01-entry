# free-model-aggregator-01-entry

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/palzzzzz/free-model-aggregator-01-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/palzzzzz/free-model-aggregator-01-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free AI model aggregator in one Python file (3.13, standard library only). It serves the OpenAI Chat Completions API and the Anthropic Messages API — JSON and streaming — over the ten free-tier providers the trial brief lists, speaking each provider's own wire (OpenAI, Anthropic, Gemini). Every request is routed to the strongest free model that meets its requirements (tools, vision, JSON mode, context length), planned against each provider's per-trial budget; when a provider refuses, the request fails over to the next candidate and the answer is returned in the wire the client used. `python3 main.py --list` prints the catalog it registers, byte for byte.

## Run it

```sh
python3 main.py --list
MAKEFUN_ENV_URL=<env url> MAKEFUN_PROMPT_FILE=<brief file> python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=n05t7x5q&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
