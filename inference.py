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
TEMPERATURE = 0.0
MAX_TOKENS = 1024
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

=== CATEGORY RULES (choose exactly one) ===
- "billing": payment issues, charges, refunds, subscription cancellation, money-back requests
- "technical": software bugs, crashes, API issues, export failures, rate limits, production outages
- "account": login problems, credentials, security, hacked accounts, data breaches, GDPR/privacy, fraud on account, phishing, personal data concerns, legal threats about personal data
- "shipping": delivery, orders, tracking, delayed shipments
- "general": ONLY use this if the ticket truly doesn't fit any of the above categories. Almost never needed.

IMPORTANT category clarifications:
- GDPR complaints, data breach reports, phishing, privacy violations = ALWAYS "account" (NOT "general")
- Legal threats about personal data or security = "account"
- Hacked account + fraudulent charges = "account" (primary issue is security, not billing)
- API/rate limit issues affecting production = "technical"
- Data loss due to system bug = "technical"

=== PRIORITY RULES (be precise) ===
- "low": purely informational, no action needed
- "medium": standard issue, no time pressure (e.g. missing order with tracking, general questions)
- "high": time-sensitive or frustrated customer, duplicate charges, login failures, cancellation requests
- "urgent": legal threats, data loss, production systems down, fraud, hacked accounts, GDPR/compliance, enterprise SLA breach, repeat contact for unresolved critical issues

IMPORTANT priority calibrations:
- Duplicate billing charge = "high" (not urgent)
- Login/credential issues = "high" (not urgent)
- Missing/delayed shipment with tracking = "medium" (not high)
- Subscription cancellation with refund = "high"
- App crash with repeat contact + business impact = "urgent"
- Data loss/deletion = "urgent"
- Legal threats, GDPR, fraud, hacked account = "urgent"
- Enterprise production outage, SLA breach = "urgent"

=== ESCALATION RULES ===
Set "escalate" to true ONLY for these scenarios:
- Legal threats or compliance issues (GDPR, lawsuits, solicitors)
- Security breaches, hacked accounts, fraud
- Data loss or accidental deletion requiring restoration
- Repeat contact where the previous resolution failed and issue is critical
- Enterprise/SLA customers with production-impacting issues
- Any ticket mentioning lawyers, legal action, regulatory complaints

Set "escalate" to false for:
- Routine billing (duplicate charges, refunds, cancellations)
- Standard login/credential issues
- Shipping inquiries
- First-contact technical issues without business-critical impact

=== TAG RULES ===
Provide 2-6 kebab-case tags. Use specific, descriptive tags such as:
- duplicate-charge, refund, cancellation, money-back-guarantee
- login, credentials, password
- tracking, delayed-shipment
- crash, export, repeat-contact, business-impact
- data-loss, restore, urgent
- gdpr, data-breach, legal, compliance, sensitive
- fraud, hacked, security, multi-issue, angry-customer
- enterprise, api, sla, production-down, rate-limit, compensation

=== RESPONSE RULES ===
Write a professional, empathetic response of at least 150 characters. Your response MUST:
- Start with a sincere apology (use words: "apologize" or "sorry")
- Directly address the customer's specific concern
- Explain what action you are taking
- Include relevant keywords from the issue domain:
  * For billing: use "refund", "billing", "apologize"
  * For technical issues: use "engineer", "sorry", and if escalating: "escalate", "urgent"
  * For account/login: use "login", "password", "account"
  * For shipping: use "order", "tracking", "shipping"
  * For security/fraud: use "secure", "fraud", "investigate", "escalate", "apologize"
  * For legal/GDPR: use "legal", "privacy", "data", "escalate", "urgent", "sorry"
  * For enterprise/SLA: use "enterprise", "sla", "restore", "compensat" (as in compensate/compensation), "escalate", "urgent", "apologize"
- If escalating, explicitly say you are escalating to a senior/specialist team
- Provide a concrete next step or timeline

Think carefully. Escalation and priority errors are heavily penalized.
""").strip()

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(
    client: OpenAI,
    ticket_subject: str,
    ticket_body: str,
    customer_history: list,
    step_feedback: str = "",
) -> dict:
    """Call the LLM and parse its JSON triage decision."""
    history_text = ""
    if customer_history:
        history_text = "\n\nCUSTOMER HISTORY (this customer has contacted before — consider escalation if previous issue was unresolved):\n" + "\n".join(
            f"- [{h.get('date','')}] {h.get('subject','')} → Resolution: {h.get('resolution','')}"
            for h in customer_history
        )

    feedback_text = ""
    if step_feedback and step_feedback != "Episode started. Good luck!":
        feedback_text = f"\n\nFEEDBACK FROM PREVIOUS TRIAGE (learn from this):\n{step_feedback}"

    user_msg = f"SUBJECT: {ticket_subject}\n\nBODY:\n{ticket_body}{history_text}{feedback_text}"

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
            "response": "We sincerely apologize for any inconvenience. We have received your request and our team will investigate this matter urgently. We will get back to you shortly with a resolution.",
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
        # Reset — pass task as query parameter to ensure correct ticket set
        obs_data = http_post(f"/reset", {})
    except Exception as e:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {"steps": 0, "rewards": [], "score": 0.0, "success": False}

    done = obs_data.get("done", False)
    step_feedback = obs_data.get("step_feedback", "")
    step = 0

    while not done and step < MAX_STEPS_PER_TASK:
        step += 1
        steps_taken = step

        # Get LLM triage decision (with feedback from previous step)
        decision = call_llm(
            client,
            ticket_subject=obs_data.get("subject", ""),
            ticket_body=obs_data.get("body", ""),
            customer_history=obs_data.get("customer_history", []),
            step_feedback=step_feedback,
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
            step_feedback = obs_data.get("step_feedback", "")
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
