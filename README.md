# EzeAi Comfy Agent

**A local AI agent that lives inside ComfyUI.** Describe what you want; it reads your
canvas, finds the right nodes and models, builds or edits the workflow, runs it, looks at
the result and improves it. It runs on language models on **your own PC** (Ollama or
llama.cpp). No cloud, no API keys, no telemetry, nothing is downloaded.

- Three modes: **Inspect** (read-only), **Ask** (every change is a card you approve),
  **Autonomous** (build → run → review → improve, with a live mission panel).
- Validated graph edits: the model cannot invent nodes or wire the wrong types.
- Sees your canvas (screenshot) and your generated images through a vision model.
- An in-graph node, **EzeAi Comfy Agent (LLM)**, for prompt writing inside workflows.
- Pure Python standard library plus what ComfyUI already ships. No `pip install` needed.

---

## Contents

1. [Requirements](#requirements)
2. [Install](#install)
3. [Set up a language model](#set-up-a-language-model)
4. [First run](#first-run)
5. [The interface](#the-interface)
6. [Modes](#modes)
7. [What the agent can do](#what-the-agent-can-do)
8. [Example prompts](#example-prompts)
9. [Settings](#settings)
10. [The in-graph LLM node](#the-in-graph-llm-node)
11. [Safety and privacy](#safety-and-privacy)
12. [Troubleshooting](#troubleshooting)
13. [Known limitations](#known-limitations)
14. [Development and tests](#development-and-tests)
15. [Contributing and license](#contributing-and-license)

---

## Requirements

| Need | Notes |
| --- | --- |
| ComfyUI | Recent build with the new frontend (tested on frontend 1.53). Windows portable, Linux or macOS. |
| A local LLM runtime | **Ollama** (easiest) **or** GGUF files run through `llama-server` (llama.cpp). |
| A tool-calling model | A 4B model is the practical minimum. See [recommended models](#recommended-models). |
| GPU memory | Enough for your image model **or** the LLM. The agent unloads the LLM before rendering. |

No extra Python packages are required.

## Install

### Option A: git clone (recommended)

Open a terminal in your ComfyUI `custom_nodes` folder.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ezeai/ComfyUI-EzeAi-Comfy-Agent.git
```

### Option B: download the zip

1. On the GitHub page click **Code → Download ZIP**.
2. Unzip it into `ComfyUI/custom_nodes/` so the path is
   `ComfyUI/custom_nodes/ComfyUI-EzeAi-Comfy-Agent/__init__.py`.
   (If the zip adds a `-main` suffix to the folder name, that is fine.)

### Option C: ComfyUI-Manager

Search for **EzeAi Comfy Agent** in ComfyUI-Manager once it is listed there.

### After installing

1. **Restart ComfyUI** (close the console window and start it again).
2. Look for this line in the console:
   `[EzeAi Comfy Agent] agent API at /ezeai/agent/`
3. Hard-refresh the browser (**Ctrl+F5**). The **✦ EzeAi Agent** tab appears in the left sidebar.

To update later: `git pull` inside the folder, restart ComfyUI, hard-refresh.
Your settings, chats and memory live in `user_data/` and are never overwritten.

## Set up a language model

The agent finds models automatically from three places. Open **⚙ Inspector → Models**
to see everything it found and what each model can do (`tools`, `vision`, `thinking`).

### Path 1: Ollama (easiest)

1. Install Ollama from <https://ollama.com> and make sure it is running.
2. Pull a model that supports tools, for example:
   ```bash
   ollama pull qwen3:4b-instruct
   ollama pull qwen3.5:9b      # also does vision
   ```
3. In ComfyUI open the agent. It uses what `ollama list` shows. Done.

### Path 2: GGUF files (llama.cpp)

1. Put `.gguf` files in any of these folders:
   - `ComfyUI/custom_nodes/ComfyUI-EzeAi-Comfy-Agent/llm/`
   - `ComfyUI/models/LLM/`
   - `ComfyUI/models/llm_gguf/`
   - any folder you add in **Inspector → Settings → Extra model folders**
2. For vision, put the matching `mmproj*.gguf` projector **next to** the model file.
3. The agent starts `llama-server` for you. By default it uses the copy bundled with
   Ollama. To use your own build set **Settings → llama-server path** to
   `llama-server` / `llama-server.exe`.

Tool support is read from the chat template inside each GGUF. A model without a tool
template is marked as such and is not used as the agent.

### Path 3: a server you already run

Run any OpenAI-compatible server on this PC (LM Studio, `llama-server`, etc.) and put
its address in **Settings → External llama.cpp server URL**, for example
`http://127.0.0.1:8080`. Servers on other machines are refused unless you tick
**Allow model servers on other machines**.

### Recommended models

| Role | Good choices | Why |
| --- | --- | --- |
| Agent (plans and calls tools) | Spark 4B, Qwen3 4B Instruct, Qwen3.5 4B/9B | Reliable tool calling, 4 to 10 s per turn on a mid-range GPU. |
| Fast, small agent | Qwen3.5 2B (Q8) | Measured 3 to 13 s per task and calls tools correctly; less reliable than a 4B on multi-step fixes. |
| Vision (looks at images and the canvas) | Qwen3.5 4B or 9B | The agent routes image questions here automatically. |

Pick models in **Inspector → Settings → Agent model / Vision model**, or leave on auto.

**Not every GGUF is a chat model.** Scoring or classifier fine-tunes (for example "Decision" models that only output yes/no scores) can't drive the agent even if they report tools. Use a normal instruct/chat model.

**Using a custom or fine-tuned model:** any Ollama model or GGUF that reports the
`tools` capability will work as the agent. A model that does not support tool calls
cannot drive the agent. It can still be used by the in-graph LLM node for plain text
tasks such as prompt writing.

## First run

1. Open the **✦ EzeAi Agent** tab (or press **Ctrl+Shift+A**).
2. The header shows the model chip and the mode chip (starts in **Ask**).
3. Click a suggestion such as *Explain this workflow*, or type a request and press Enter.
4. Right-click any node → **Ask EzeAi Agent about this node**.

## The interface

| Part | What it does |
| --- | --- |
| **Model chip** | Shows the active model. Click to open the Inspector. |
| **Mode chip** | Click to cycle Inspect → Ask → Autonomous. |
| **+ / ☰ / ⚙** | New chat, chat history, Inspector. |
| **Mission panel** | Appears in Autonomous mode: goal, Plan → Build → Run → Review stepper, cycle counter, elapsed time, live render progress, step log, **Stop** button. |
| **Activity fold** | Under each answer: every tool the agent used, with arguments and results. |
| **Cards** | Proposed changes with an exact diff and **Apply / Dismiss / Show**, then **Undo**. |
| **Composer** | Type, attach or paste images (📎), **⏏** frees VRAM by unloading the LLM. |
| **Inspector** | Models and their capabilities, runtime state, settings, memory, and the exact context sent to the model. |

## Modes

| Mode | What the agent may do |
| --- | --- |
| **Inspect** | Read-only. It reads, explains, diagnoses and looks. Change tools are never offered to the model. |
| **Ask** (default) | Every change is a card with the exact edits. Nothing happens until you click. |
| **Autonomous** | Applies edits immediately (with Undo), queues the run, then reviews the image against your request and either stops or makes one improvement, up to *N* cycles (default 3). |

In **every** mode, removing nodes you already had, replacing the canvas, and overwriting a
saved workflow always wait for your click. This rule is enforced in code
(`agent/policy.py`), not in a prompt.

## What the agent can do

**Look (runs automatically):** read the canvas, search installed nodes, read a node's
exact inputs and allowed values, list model files and saved workflows, check the last run
and its exact error, look at an output image, look at the canvas screenshot, check GPU
memory, load a skill playbook, remember a preference.

**Do (always validated, a card first unless Autonomous):**
- **Create:** `create_z_image_turbo`, `create_checkpoint_t2i`, `create_upscale_image`. Offered only when their nodes and models are installed.
- **Customize:** `set_widgets`, `add_node`, `connect`, `disconnect`, `remove_node`, `bypass_node`, and `insert_after` (places a LoRA, patch or filter between two nodes and rewires every consumer).
- **Run and manage:** `queue_run` (frees VRAM first), `open_workflow`, `save_workflow`.

All edits in one turn become **one** card, re-checked against your canvas at the moment
you click Apply.

**"Computer use" scope:** the agent sees the ComfyUI canvas and acts only through
validated graph edits, workflow open/save and the queue. It does **not** control your
mouse or keyboard or other applications, by design.

## Example prompts

```
Explain this workflow in plain English.
Why did my last run fail?
Look at my canvas. What is wrong?
Make a photoreal portrait of a lighthouse keeper at dusk.
Set 30 steps and cfg 6 on my sampler.
Add a LoRA between the checkpoint loader and the sampler.
Upscale the last image 2x.
Save this workflow as "Portrait base".
```

In Autonomous mode give a goal, not steps:
`Make a cozy cabin in a snowy forest at night.`

## Settings

Open **⚙ Inspector → Settings**. Stored in `user_data/settings.json`.

| Setting | Default | Meaning |
| --- | --- | --- |
| Agent model / Vision model | auto | Which models to use. |
| Mode | ask | inspect, ask or auto. |
| Autonomous cycles | 3 | Max build → run → review rounds per request. |
| Let the agent see the canvas | on | Sends a downscaled canvas screenshot with each message (stays on this PC). |
| Ollama URL | `http://127.0.0.1:11434` | Where Ollama listens. |
| llama-server path | bundled with Ollama | Your own `llama-server` build. |
| External server URL | empty | An already running OpenAI-compatible server. |
| Extra model folders | none | More places to scan for GGUF files. |
| Allow model servers on other machines | off | Remote hosts are refused unless on. |
| Max tool rounds | 8 | Tool calls per turn. |
| Context tokens | 16384 | Context window requested. |
| Max tokens per reply | 1536 | Hard cap against runaway output. |
| Run timeout | 240 s | A turn is stopped after this long. |
| Keep alive | 2m | How long a model stays loaded when idle. |
| Free VRAM before queue | on | Unloads the LLM before a render. |

## The in-graph LLM node

Add **EzeAi Comfy Agent (LLM)** (category *EzeAi / Comfy Agent*) to use a local model as a
workflow step.

| Task | Use |
| --- | --- |
| `prompt_writer` | Turn an idea into a rich image prompt. |
| `prompt_improver` | Rewrite an existing prompt. |
| `describe_image` | Describe an IMAGE input with a vision model. |
| `chat` | General text. |
| `json` | Structured output. |

Outputs `text` (wire into a CLIPTextEncode) and `info`. **unload_after** (on by default)
frees VRAM for the sampler that follows.

## Safety and privacy

- **Local-first:** model servers not on this PC are refused unless you allow them. No
  telemetry, no analytics, no update checks, no outbound requests from this node.
- **Validated edits:** unknown node types, mismatched socket types and out-of-range values are rejected and sent back to the model to fix.
- **Evidence before claims:** if the agent says it did something but called no tool, the loop asks it to act or correct itself, and the UI shows **⚠ No action was taken**.
- **Bounded:** token caps, repetition cut-off, per-turn time limit, a stuck-loop guard and a cycle limit.
- **Paid API nodes** are hidden from search and cannot be added.
- Your chats, memory and settings stay in `user_data/` on disk.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| No **EzeAi Agent** tab | Restart ComfyUI, then **Ctrl+F5**. Check the console for `[EzeAi Comfy Agent]` and any traceback. |
| Chip says **offline** | The agent API did not load. Read the console traceback; make sure the folder name has no spaces problem and `__init__.py` is directly inside it. |
| **No local model found** | Start Ollama (`ollama list` must work) or put a `.gguf` in `llm/`. Open the Inspector to see what was scanned. |
| Agent answers but never makes cards | The selected model lacks `tools`. Choose a tool-capable model in Inspector → Settings. |
| Says "Done" but nothing changed | Small models over-claim in wording. Trust the card status and the "changes waiting for you" line. |
| Out of memory while rendering | Keep **Free VRAM before queue** on, or use a smaller agent model. Use **⏏** to unload manually. |
| Vision is very slow | A diffusion model is holding VRAM. Use a smaller vision model such as Qwen3.5 4B. |
| Remote host refused | Intended. Tick **Allow model servers on other machines** only if you trust that server. |
| Run did not start | Click **Run** on the card. In Autonomous mode check the mission panel's note for the reason. |
| Reset everything | Close ComfyUI and delete `user_data/`. |

## Known limitations

- Small models over-claim in wording, and the automatic image review is generous. Treat it as a second opinion.
- Hand-wiring large graphs from scratch is the weakest skill of a 4B model, which is why recipes and `insert_after` exist.
- Ready-made recipes cover text-to-image (Z-Image Turbo, checkpoint) and upscaling. Video, ControlNet and inpainting graphs can be edited node by node but have no recipe.
- Subgraphs are read as single nodes and are not edited inside.
- The built-in recipes expect specific model files (for example `z_image_turbo_bf16`). If they are missing the recipe is simply not offered.

## Development and tests

```bash
cd ComfyUI/custom_nodes/ComfyUI-EzeAi-Comfy-Agent
python -m unittest discover -s tests -p "test_*.py"
```

The tests run without ComfyUI or a model (a fixture of 669 built-in nodes is included).
Layout:

```
__init__.py            registration (never breaks ComfyUI if something fails)
api.py                 /ezeai/agent/* routes, streams over ComfyUI's websocket
nodes.py               EzeAi Comfy Agent (LLM) node
agent/config.py        settings
agent/models.py        model discovery + GGUF metadata reader
agent/runtime.py       llama-server lifecycle
agent/providers.py     one chat() over Ollama / llama.cpp / OpenAI-compatible
agent/comfy_bridge.py  reads nodes, models, queue and history
agent/graph_ops.py     validated graph-edit language
agent/recipes.py       proven workflow builders
agent/tools.py         look and do tools
agent/policy.py        what may auto-apply (code, not prompt)
agent/loop.py          the agent loop and guards
agent/memory.py        preferences and skills
agent/skills/*.md      playbooks loaded on demand
web/comfy_agent.js     sidebar UI
web/comfy_agent.css    styling
llm/                   drop GGUF files here
user_data/             settings, chats, memory (git-ignored)
```

## Contributing and license

Issues and pull requests are welcome. Please run the tests first and keep changes small
and focused. To publish or fork this project see [PUBLISHING.md](PUBLISHING.md).

Released under the **MIT License** (see [LICENSE](LICENSE)).
