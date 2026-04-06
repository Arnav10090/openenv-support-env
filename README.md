---
title: Customer Support Triage Environment
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - nlp
  - agent-evaluation
license: mit
---

# 🎫 Customer Support Triage — OpenEnv Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HF-Space-yellow)](https://huggingface.co/spaces/Arnav100904/customer-support-triage)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **production-ready OpenEnv environment** where an AI agent must triage customer support tickets — the same task real support teams do every day.

---

## 🎯 What Is This?

In real support operations, agents must:
1. **Classify** tickets (billing, technical, account, shipping, general)
2. **Prioritize** urgency (low → medium → high → urgent)
3. **Decide escalation** — does this need a human?
4. **Draft a response** — professional, empathetic, correct
5. **Tag** tickets for routing and analytics

This environment lets an RL agent or LLM learn and be evaluated on all five sub-tasks, with shaped rewards that provide signal throughout the episode — not just at the end.

---

## 🧩 Action Space

Each step the agent submits a `SupportAction`:

| Field | Type | Values | Description |
|---|---|---|---|
| `category` | `str` | `billing` \| `technical` \| `account` \| `shipping` \| `general` | Ticket classification |
| `priority` | `str` | `low` \| `medium` \| `high` \| `urgent` | Urgency level |
| `escalate` | `bool` | `true` / `false` | Whether to escalate to human agent |
| `response` | `str` | Any string (≥80 chars recommended) | Draft reply to customer |
| `tags` | `List[str]` | e.g. `["refund", "duplicate-charge"]` | Labels for routing |

---

## 👁️ Observation Space

Each step the agent receives a `SupportObservation`:

| Field | Type | Description |
|---|---|---|
| `ticket_id` | `str` | Unique ticket identifier |
| `subject` | `str` | Ticket subject line |
| `body` | `str` | Full ticket message |
| `customer_history` | `List[dict]` | Previous tickets from the same customer |
| `available_categories` | `List[str]` | Valid category options |
| `available_priorities` | `List[str]` | Valid priority options |
| `step_feedback` | `str` | Feedback from the last action |
| `reward` | `float` | Reward received for last action |
| `done` | `bool` | Whether the episode is complete |

---

## 📋 Tasks

### 🟢 Easy — Clear-cut triage
- **Tickets**: Duplicate billing charge, login failure, missing shipment
- **Key skills**: Category & priority classification
- **Grading**: category (40%) + priority (30%) + response (20%) + tags (10%)
- **Expected score**: 0.7–0.9 for a capable LLM

### 🟡 Medium — Ambiguous escalation
- **Tickets**: Repeat-contact crashes, cancellation + refund, data deletion
- **Key skills**: Escalation judgment, tone, customer history awareness
- **Grading**: category (25%) + priority (25%) + escalation (25%) + response (20%) + tags (5%)
- **Expected score**: 0.5–0.75 for a capable LLM

### 🔴 Hard — Complex multi-issue tickets
- **Tickets**: GDPR/legal threats, account fraud + hacked, enterprise API SLA breach
- **Key skills**: All of the above + legal sensitivity, multi-issue response, security awareness
- **Grading**: category (15%) + priority (20%) + escalation (30%) + response (25%) + tags (10%)
- **Expected score**: 0.35–0.60 even for frontier models

---

## 🏆 Reward Function

Rewards are **shaped over the full trajectory** — not just binary end-of-episode:

- **Category** — binary (correct = 1.0, wrong = 0.0)
- **Priority** — partial credit by proximity (exact = 1.0, off-by-1 = 0.5, off-by-2 = 0.25)
- **Escalation** — binary (correct decision = 1.0, wrong = 0.0)
- **Response quality** — keyword coverage + length adequacy
- **Tags** — fraction of expected tags present
- **Penalties** — empty response (−0.3), invalid category (−0.1), invalid priority (−0.1)

All component scores are weighted per-difficulty and combined into a scalar reward in **[0.0, 1.0]**.

---

## 🚀 Quick Start

### Option 1: Use the live HF Space

```python
import json, urllib.request

BASE_URL = "https://arnav100904-customer-support-triage.hf.space"

# Reset
obs = json.loads(urllib.request.urlopen(
    urllib.request.Request(f"{BASE_URL}/reset", data=b"{}", 
    headers={"Content-Type":"application/json"}, method="POST")
).read())

print(obs["subject"])  # First ticket subject

# Step
action = {
    "category": "billing", "priority": "high",
    "escalate": False, "tags": ["refund"],
    "response": "We sincerely apologize for the duplicate charge. We will process your refund within 3-5 business days.",
    "metadata": {}
}
result = json.loads(urllib.request.urlopen(
    urllib.request.Request(f"{BASE_URL}/step", 
    data=json.dumps(action).encode(),
    headers={"Content-Type":"application/json"}, method="POST")
).read())
print(f"Reward: {result['reward']:.2f}")
```

### Option 2: Run locally with Docker

```bash
# Pull from HF registry
docker pull registry.hf.space/arnav100904-customer-support-triage:latest

# Run easy task
docker run -p 7860:7860 -e SUPPORT_TASK=easy registry.hf.space/arnav100904-customer-support-triage:latest

# Run hard task
docker run -p 7860:7860 -e SUPPORT_TASK=hard registry.hf.space/arnav100904-customer-support-triage:latest
```

### Option 3: Build from source

```bash
git clone https://github.com/Arnav100904/openenv-support-env
cd openenv-support-env

# Build
docker build -t support-env .

# Run
docker run -p 7860:7860 -e SUPPORT_TASK=medium support-env

# Health check
curl http://localhost:7860/health
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check → `{"status": "healthy"}` |
| `/reset` | POST | Start new episode, return first ticket |
| `/step` | POST | Submit triage action, get next ticket + reward |
| `/state` | GET | Episode metadata and running scores |
| `/docs` | GET | Auto-generated OpenAPI docs |

---

## 🏃 Running the Baseline Inference Script

```bash
# Set required environment variables
export HF_TOKEN=hf_your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENVIRONMENT_URL=https://arnav100904-customer-support-triage.hf.space

# Run baseline
python inference.py
```

### Baseline Scores (Qwen2.5-72B-Instruct)

| Task | Score | Notes |
|---|---|---|
| Easy | 0.78 | Strong category + priority classification |
| Medium | 0.61 | Occasional escalation errors |
| Hard | 0.44 | Legal/enterprise sensitivity challenging |
| **Overall** | **0.61** | Solid baseline for a frontier model |

---

## 📁 Project Structure

```
openenv-support-env/
├── inference.py              ← Baseline inference script (mandatory)
├── openenv.yaml              ← OpenEnv manifest
├── Dockerfile                ← Container definition
├── requirements.txt
├── pyproject.toml
├── README.md
├── server/
│   └── app.py               ← FastAPI server
└── src/
    └── envs/
        └── support_env/
            ├── __init__.py
            ├── models.py    ← Action / Observation / State types
            ├── environment.py ← Core logic
            ├── grader.py    ← Deterministic scoring
            ├── tickets.py   ← Ticket dataset with ground truth
            └── client.py    ← HTTP client
```

---

## 🎖️ Why Customer Support Triage?

Support triage is a high-stakes real-world task that:
- Happens millions of times per day across every industry
- Has measurable, deterministic correct answers (category, priority, escalation)
- Has natural difficulty gradations (simple billing → complex legal/fraud)
- Tests multiple agent skills: classification, reasoning, writing, judgment
- Has direct business value if solved well by AI agents

It's a domain where RL agents trained in this environment could realistically replace or augment real workflows — making it ideal for OpenEnv's mission.

---

## 📜 License

MIT License — see [LICENSE](LICENSE).

Built for the **Meta × Scaler OpenEnv Hackathon, Round 1, April 2026**.
