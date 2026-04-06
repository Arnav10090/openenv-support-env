"""
FastAPI server for the Customer Support Triage Environment.

SUPPORT_TASK env var sets default task (easy|medium|hard).
Pass ?task=medium to /reset to switch without restarting.

Endpoints:
  GET  /health           – Health check
  POST /reset?task=easy  – Start new episode (optional task param)
  POST /step             – Submit triage action, get reward + next obs
  GET  /state            – Episode metadata and running stats
  GET  /docs             – Auto-generated OpenAPI docs
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from envs.support_env.environment import SupportEnvironment
from envs.support_env.models import SupportAction

DEFAULT_TASK = os.getenv("SUPPORT_TASK", "easy")

app = FastAPI(
    title="Customer Support Triage — OpenEnv",
    description=(
        "An OpenEnv-compatible RL environment where an AI agent triages customer "
        "support tickets: classify, prioritise, escalate, and draft a reply.\n\n"
        "Set `SUPPORT_TASK` env var (easy|medium|hard) or pass `?task=` to `/reset`."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[SupportEnvironment] = None
_current_task: str = DEFAULT_TASK


# ── Schemas ────────────────────────────────────────────────────────────────

class ActionRequest(BaseModel):
    category: str = Field("general", description="billing|technical|account|shipping|general")
    priority: str = Field("medium", description="low|medium|high|urgent")
    response: str = Field("", description="Draft reply to the customer (aim for ≥80 chars)")
    escalate: bool = Field(False, description="Escalate to a human agent?")
    tags: List[str] = Field(default_factory=list, description="Routing tags")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ObservationOut(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_history: List[Dict[str, Any]]
    available_categories: List[str]
    available_priorities: List[str]
    step_feedback: str
    reward: float
    done: bool
    metadata: Dict[str, Any]


class StepOut(BaseModel):
    observation: ObservationOut
    reward: float
    done: bool
    info: Dict[str, Any]


class StateOut(BaseModel):
    episode_id: str
    task_name: str
    step_count: int
    total_reward: float
    tickets_processed: int
    correct_categories: int
    correct_priorities: int
    escalations_made: int
    escalations_needed: int
    metadata: Dict[str, Any]


# ── Helpers ────────────────────────────────────────────────────────────────

def _obs_to_dict(obs) -> Dict[str, Any]:
    return {
        "ticket_id": obs.ticket_id,
        "subject": obs.subject,
        "body": obs.body,
        "customer_history": obs.customer_history,
        "available_categories": obs.available_categories,
        "available_priorities": obs.available_priorities,
        "step_feedback": obs.step_feedback,
        "reward": obs.reward,
        "done": obs.done,
        "metadata": obs.metadata,
    }


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "task": _current_task, "version": "1.0.0"}


@app.post("/reset", response_model=ObservationOut)
def reset(task: Optional[str] = Query(default=None, description="easy | medium | hard")):
    """Reset the environment. Optionally pass ?task=easy|medium|hard."""
    global _env, _current_task
    chosen = (task or DEFAULT_TASK).strip().lower()
    if chosen not in ("easy", "medium", "hard"):
        raise HTTPException(400, f"Invalid task {chosen!r}. Use easy, medium, or hard.")
    _current_task = chosen
    _env = SupportEnvironment(task=chosen)
    obs = _env.reset()
    return _obs_to_dict(obs)


@app.post("/step", response_model=StepOut)
def step(req: ActionRequest):
    """Submit a triage action. Call /reset first."""
    global _env
    if _env is None or not _env._episode_started:
        raise HTTPException(400, "No active episode — call /reset first.")
    sa = SupportAction(
        category=req.category,
        priority=req.priority,
        response=req.response,
        escalate=req.escalate,
        tags=req.tags,
        metadata=req.metadata,
    )
    obs, reward, done, info = _env.step(sa)
    return {"observation": _obs_to_dict(obs), "reward": reward, "done": done, "info": info}


@app.get("/state", response_model=StateOut)
def state():
    """Current episode state and running statistics."""
    global _env
    if _env is None:
        raise HTTPException(400, "No active episode — call /reset first.")
    s = _env.state
    return {
        "episode_id": s.episode_id,
        "task_name": s.task_name,
        "step_count": s.step_count,
        "total_reward": s.total_reward,
        "tickets_processed": s.tickets_processed,
        "correct_categories": s.correct_categories,
        "correct_priorities": s.correct_priorities,
        "escalations_made": s.escalations_made,
        "escalations_needed": s.escalations_needed,
        "metadata": s.metadata,
    }
