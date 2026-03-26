# Meta-Harness

Agent scaffold for [Terminal-Bench 2.0](https://tbench.ai), built on top of [Terminus-KIRA](https://github.com/krafton-ai/KIRA) by KRAFTON AI and [Harbor](https://github.com/laude-institute/harbor)'s Terminus-2 framework.

## Results

76.4% on Terminal-Bench 2.0 (89 tasks × 5 trials, Claude Opus 4.6).

| Split  | N  | Score |
|--------|---:|------:|
| Easy   |  4 | 100.0 |
| Medium | 55 |  81.1 |
| Hard   | 30 |  64.7 |
| **All**| 89 |**76.4**|

## Usage

```bash
pip install harbor

export ANTHROPIC_API_KEY=<your-key>

harbor run \
  --agent-import-path agent:AgentHarness \
  -d terminal-bench@2.0 \
  -m anthropic/claude-opus-4-6 \
  -e runloop \
  -n 20 \
  --n-attempts 5
```

## Method

Meta-Harness extends the [Terminus-KIRA](https://github.com/krafton-ai/KIRA) agent with environment bootstrapping: before the agent loop starts, it gathers a snapshot of the sandbox environment (working directory, file listing, available languages/tools, package managers, memory) and injects it into the initial prompt. This saves 2-5 early exploration turns that the agent normally spends on `ls`, `which python3`, etc.

The agent was discovered through automated harness evolution. More details coming soon.

## Acknowledgements

We thank KRAFTON AI for compute support.
