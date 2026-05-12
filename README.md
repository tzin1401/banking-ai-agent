# Banking AI-Agent

> Project 3 — *Applications of Natural Language Processing in Industry*
> Faculty of Information Technology, University of Science, VNU-HCM
> Lecturer: Dr. Nguyen Hong Buu Long — 04/2026

A small **agentic workflow** for banking customer support. Given a customer
message, the system detects the intent, retrieves relevant policy, drafts a
reply, validates it, and decides whether to send the reply, ask for more
information, or escalate the case to a human agent.



---

## Workflow

```
customer_msg
     │
     ▼
┌─────────────────┐   ┌──────────────────┐   ┌────────────────┐
│  Intent Node    │ → │  Priority Node   │ → │  Policy Node   │
│  (Lab 2 model)  │   │  (rules)         │   │  (lookup)      │
└─────────────────┘   └──────────────────┘   └────────┬───────┘
                                                      │
              ┌───────────────────────────────────────┘
              ▼
       ┌────────────────┐   ┌──────────────────┐   ┌───────────────┐
       │  Draft Node    │ → │  Validation Node │ → │  Router Node  │
       │  (Ollama LLM)  │   │  (rules)         │   │  (decision)   │
       └────────────────┘   └──────────────────┘   └───────┬───────┘
                                                           │
                                                           ▼
                                           reply / ask_more / escalate
```

| Node | Implementation | Notes |
|------|----------------|-------|
| Intent Detection | Fine-tuned **Llama-3.2-3B + LoRA** (from [Lab 2](https://github.com/tzin1401/banking-intent-unsloth)) | Test accuracy **93.70%** on BANKING77 |
| Priority / Risk | Rule-based (keywords + intent) | low / medium / high |
| Policy Retrieval | Static dict lookup | 77 BANKING77 intents → policy snippets |
| Response Drafting | **`qwen2.5:7b`** via Ollama (HTTP) | T4-friendly default; switch `OLLAMA_MODEL=gpt-oss:20b` on A100/L4 |
| Validation | Rule-based (length, placeholders, confidence) | |
| Routing | Rule-based combining all prior signals | |

### Why qwen2.5:7b instead of gpt-oss:20b?

The lab handout *encourages* `gpt-oss:20b`, but that model alone needs ≥16 GB VRAM. Combined with the Lab 2 Unsloth-3B intent classifier (~2.5 GB) it OOMs a Colab Free T4 (15.36 GB usable). The orchestrator talks to **any Ollama chat model** via HTTP, so we default to `qwen2.5:7b` (~4.7 GB) for reliability and document the upgrade path for users with bigger GPUs.

---

## Architecture

Everything is designed to run inside **one Google Colab notebook**:

```
┌─────────────────── Colab GPU runtime ───────────────────┐
│                                                         │
│   FastAPI (port 8000) ──┐                               │
│       │ orchestrator    │ HTTP                          │
│       ▼                 ▼                               │
│   IntentClassifier      Ollama (localhost:11434)        │
│   (Unsloth + LoRA)      gpt-oss:20b                     │
│                                                         │
└────────────────────┬────────────────────────────────────┘
                     │  Pinggy SSH tunnel
                     ▼
              http://<id>.a.free.pinggy.link
```

Only the FastAPI port is exposed publicly; Ollama is reached from the
orchestrator over `localhost`.

---

## Repository layout

```
banking-agent/
├── README.md
├── requirements.txt
├── run.py                  # Launch FastAPI (no business logic)
├── app/
│   ├── main.py             # FastAPI app + routes
│   ├── core/
│   │   ├── settings.py
│   │   └── schemas.py
│   ├── data/
│   │   └── policies.py
│   ├── clients/
│   │   ├── base.py
│   │   └── ollama_client.py
│   ├── nodes/
│   │   ├── intent_node.py
│   │   ├── priority_node.py
│   │   ├── policy_node.py
│   │   ├── draft_node.py
│   │   ├── validation_node.py
│   │   └── router_node.py
│   └── agent/
│       └── orchestrator.py
├── examples/
│   └── sample_requests.json
└── notebooks/
    └── colab_run.ipynb
```

---

## Quick start — local mock mode

Run the full pipeline **without** a GPU using mock implementations of the
intent classifier and the LLM. Useful for developing the orchestration logic.

```bash
pip install fastapi uvicorn pydantic pydantic-settings pyyaml requests httpx python-dotenv

export INTENT_MODE=mock
export MOCK_LLM=1
python run.py
```

Then in another terminal:

```bash
curl -s -X POST http://localhost:8000/process \
     -H "Content-Type: application/json" \
     -d '{"message": "I lost my card, what should I do?"}' | jq
```

---

## Full mode (Colab)

See [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb). High-level steps:

1. Install Ollama and `ollama pull gpt-oss:20b`.
2. Clone this repo, install `requirements.txt` + Unsloth.
3. Mount Google Drive, unzip
   `MyDrive/banking-ai-agent/intent_artifacts.zip` into the project root.
4. `python run.py` to start FastAPI on port 8000.
5. Open Pinggy tunnel: `ssh -p 443 -R0:localhost:8000 qr@a.pinggy.io`.
6. Demo with the public URL using `examples/sample_requests.json`.

---

## Configuration

Environment variables (see `app/core/settings.py` for defaults):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama HTTP endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model used for drafting (switch to `gpt-oss:20b` on A100/L4) |
| `INTENT_MODE` | `unsloth` | `unsloth` (real) or `mock` (rule-based) |
| `INTENT_CONFIG_PATH` | `./configs/inference.yaml` | Lab 2 inference config |
| `INTENT_LABELS_PATH` | `./sample_data/labels.txt` | 77 BANKING77 labels |
| `MOCK_LLM` | `0` | Set to `1` to skip Ollama and return a canned draft |

---

## Video demo

🎬 [Video demo on Google Drive](https://drive.google.com/drive/u/0/folders/1hAjImvpUAM_c40VrPCCcoJCSGRS7wF3r)

---

## Related work

- **Lab 2 — Intent classifier**: <https://github.com/tzin1401/banking-intent-unsloth>
- **Dataset**: [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
