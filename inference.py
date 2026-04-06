"""
Baseline Inference Script — Customer Support Triage Environment
===============================================================
MANDATORY VARIABLES (set in environment before running):
    API_BASE_URL      The API endpoint for the LLM  (default: HF router)
    MODEL_NAME        The model identifier           (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN          Your Hugging Face / API key
    LOCAL_IMAGE_NAME  Docker image name (optional, only if using from_docker_image)

STDOUT FORMAT (strictly followed):
    [START] task=<task_name> env=customer_support_triage model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Run:
    HF_TOKEN=hf_xxx python inference.py
"""

import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment, with safe defaults
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
ENVIRONMENT_URL: str = os.getenv("ENVIRONMENT_URL", "http://localhost:7860")

TASKS = ["easy", "medium", "hard"]
MAX_STEPS_PER_TASK = 3
TEMPERATURE = 0.3
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5   # avg reward >= 0.5 → success

BENCHMARK = "customer_support_triage"

VALID_CATEGORIES = ["billing", "technical", "account", "shipping", "general"]
VALID_PRIORITIES  = ["low", "medium", "high", "urgent"]

# ---------------------------------------------------------------------------
# Logging helpers  (strict format — do NOT change field names/order)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert customer support triage specialist.
You will receive a customer support ticket and must respond with a JSON object containing your triage decision.

RESPOND ONLY WITH VALID JSON. No prose, no markdown, no explanation.

Your JSON must have exactly these fields:
{
  "category": "<one of: billing, technical, account, shipping, general>",
  "priority": "<one of: low, medium, high, urgent>",
  "escalate": <true or false>,
  "tags": ["<tag1>", "<tag2>"],
  "response": "<your full draft reply to the customer>"
}

TRIAGE GUIDELINES:
- category: billing=payment/charges/refunds, technical=bugs/crashes/API, account=login/security/data,
            shipping=delivery/orders, general=everything else
- priority: low=informational, medium=standard issue, high=time-sensitive/frustrated customer,
            urgent=legal threats / data loss / production down / fraud
- escalate: true if the issue requires a human agent (legal, security breach, unresolved repeat contact,
            enterprise SLA, data loss). false for routine issues.
- tags: 1-5 short kebab-case tags describing the issue
- response: a professional, empathetic reply. At minimum 80 characters. Address the customer's concern directly.

Think carefully. Escalation errors on hard tickets are heavily penalized.
""").strip()

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, ticket_subject: str, ticket_body: str, customer_history: list) -> dict:
    """Call the LLM and parse its JSON triage decision."""
    history_text = ""
    if customer_history:
        history_text = "\n\nCUSTOMER HISTORY:\n" + "\n".join(
            f"- [{h.get('date','')}] {h.get('subject','')} → {h.get('resolution','')}"
            for h in customer_history
        )

    user_msg = f"SUBJECT: {ticket_subject}\n\nBODY:\n{ticket_body}{history_text}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {
            "category": "general",
            "priority": "medium",
            "escalate": False,
            "tags": [],
            "response": "Thank you for contacting us. We have received your request and will get back to you shortly.",
        }

# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task: str) -> dict:
    """
    Run a single task episode against the live environment.
    Returns dict with steps, rewards, score, success.
    """
    import urllib.request
    import urllib.parse

    base_url = ENVIRONMENT_URL

    def http_post(path: str, payload: dict) -> dict:
        url = f"{base_url}{path}?task={task}"  # pass task as query param for reset
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def http_post_plain(path: str, payload: dict) -> dict:
        url = f"{base_url}{path}"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    log_start(task=task, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    error_msg = None

    try:
        # Reset — set task via env var on server, reset just starts episode
        obs_data = http_post_plain("/reset", {})
    except Exception as e:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {"steps": 0, "rewards": [], "score": 0.0, "success": False}

    done = obs_data.get("done", False)
    step = 0

    while not done and step < MAX_STEPS_PER_TASK:
        step += 1
        steps_taken = step

        # Get LLM triage decision
        decision = call_llm(
            client,
            ticket_subject=obs_data.get("subject", ""),
            ticket_body=obs_data.get("body", ""),
            customer_history=obs_data.get("customer_history", []),
        )

        # Clamp to valid values
        category = decision.get("category", "general")
        if category not in VALID_CATEGORIES:
            category = "general"
        priority = decision.get("priority", "medium")
        if priority not in VALID_PRIORITIES:
            priority = "medium"

        action_payload = {
            "category": category,
            "priority": priority,
            "escalate": bool(decision.get("escalate", False)),
            "tags": decision.get("tags", []),
            "response": decision.get("response", ""),
            "metadata": {},
        }

        # Compact action string for [STEP] log
        action_str = (
            f"category={category},priority={priority},"
            f"escalate={str(action_payload['escalate']).lower()}"
        )

        try:
            result = http_post_plain("/step", action_payload)
            reward = float(result.get("reward", 0.0))
            done = result.get("done", False)
            obs_data = result.get("observation", {})
            error_msg = None
        except Exception as e:
            reward = 0.0
            done = True
            error_msg = str(e)

        rewards.append(reward)
        log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

    score = sum(rewards) / MAX_STEPS_PER_TASK if MAX_STEPS_PER_TASK > 0 else 0.0
    score = min(max(score, 0.0), 1.0)
    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"steps": steps_taken, "rewards": rewards, "score": score, "success": success}

# ---------------------------------------------------------------------------
# Main — run all 3 tasks
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY environment variable not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = []
    for task in TASKS:
        result = run_task(client, task)
        all_scores.append(result["score"])

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n[SUMMARY] overall_score={overall:.3f} tasks={','.join(TASKS)}", flush=True)


if __name__ == "__main__":
    main()
