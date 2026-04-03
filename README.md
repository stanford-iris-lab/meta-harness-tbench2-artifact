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

### CODEX / OpenAI-compatible endpoint

If you want to run against a CODEX-style OpenAI-compatible wrapper instead of Anthropic,
set a custom base URL and keep the model name on the normal OpenAI path:

```bash
pip install harbor

export CODEX_API_BASE=https://codex-openai-wrapper/v1
export CODEX_API_KEY=${CODEX_API_KEY:-x}

harbor run \
  --agent-import-path agent:AgentHarness \
  -d terminal-bench@2.0 \
  -m openai/gpt-5.4 \
  -e runloop \
  -n 20 \
  --n-attempts 5
```

`CODEX_API_BASE` may point either to the API base (for example `/v1`) or directly to
`/chat/completions`; the harness normalizes both forms. The endpoint must support
OpenAI-style chat completions with tool calling, and `image_read` works only if the
endpoint also accepts OpenAI-style multimodal `image_url` inputs. A `codex/<model>`
alias is still supported, but `openai/<model>` avoids LiteLLM model-info warnings.

If your wrapper uses a self-signed or locally issued certificate, set:

```bash
export CODEX_SKIP_SSL_VERIFY=true
```

For repeated local runs, the repo also includes a small wrapper script plus env template:

```bash
cp .env.codex.example .env.codex
./scripts/run-codex.sh
```

The script auto-loads `.env.codex` if present and otherwise falls back to sane defaults.
By default it runs the dataset-level `hello-world@1.0` smoke test without forcing a task.
You can override the task ad hoc:

```bash
./scripts/run-codex.sh -d terminal-bench@2.0 -t <task-name> -n 1
```

## Method

Meta-Harness extends the [Terminus-KIRA](https://github.com/krafton-ai/KIRA) agent with environment bootstrapping: before the agent loop starts, it gathers a snapshot of the sandbox environment (working directory, file listing, available languages/tools, package managers, memory) and injects it into the initial prompt. This saves 2-5 early exploration turns that the agent normally spends on `ls`, `which python3`, etc.

The agent was discovered through automated harness evolution. More details coming soon.

## Acknowledgements

We thank KRAFTON AI for compute support.
