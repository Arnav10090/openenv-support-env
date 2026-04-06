"""
Deterministic graders for the Customer Support Triage Environment.

Each grader returns a float in [0.0, 1.0] and a breakdown dict.

Scoring breakdown (weights differ per difficulty):
  - category_score  : correct category chosen
  - priority_score  : correct priority chosen
  - escalation_score: correct escalation decision
  - response_score  : response quality (keyword coverage + length)
  - tag_score       : relevant tags included
  - penalty         : deductions for bad behaviour

Easy   : category(40%) + priority(30%) + response(20%) + tags(10%)
Medium : category(25%) + priority(25%) + escalation(25%) + response(20%) + tags(5%)
Hard   : category(15%) + priority(20%) + escalation(30%) + response(25%) + tags(10%)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .models import SupportAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {"billing", "technical", "account", "shipping", "general"}
VALID_PRIORITIES = {"low", "medium", "high", "urgent"}

PRIORITY_DISTANCE = {
    "low": 0, "medium": 1, "high": 2, "urgent": 3
}


def _category_score(action: SupportAction, ticket: Dict[str, Any]) -> float:
    """Binary: correct category = 1.0, wrong = 0.0."""
    cat = (action.category or "").strip().lower()
    return 1.0 if cat == ticket["gt_category"] else 0.0


def _priority_score(action: SupportAction, ticket: Dict[str, Any]) -> float:
    """
    Partial credit based on proximity:
      exact = 1.0, off by 1 = 0.5, off by 2 = 0.25, off by 3+ = 0.0
    """
    pred = (action.priority or "medium").strip().lower()
    gt = ticket["gt_priority"]
    dist = abs(PRIORITY_DISTANCE.get(pred, 1) - PRIORITY_DISTANCE.get(gt, 1))
    if dist == 0:
        return 1.0
    elif dist == 1:
        return 0.5
    elif dist == 2:
        return 0.25
    return 0.0


def _escalation_score(action: SupportAction, ticket: Dict[str, Any]) -> float:
    """Binary: correct escalation decision = 1.0, wrong = 0.0."""
    return 1.0 if action.escalate == ticket["gt_escalate"] else 0.0


def _response_score(action: SupportAction, ticket: Dict[str, Any]) -> float:
    """
    Score based on:
      - keyword coverage: how many gt_response_keywords appear in response (60%)
      - length adequacy: too short (<50 chars) = 0.3 penalty (40%)
    """
    response = (action.response or "").lower()
    keywords: List[str] = ticket.get("gt_response_keywords", [])

    # keyword coverage
    if keywords:
        hits = sum(1 for kw in keywords if kw.lower() in response)
        kw_score = hits / len(keywords)
    else:
        kw_score = 0.5  # no keywords defined → neutral

    # length
    length_score = 1.0 if len(response) >= 80 else (len(response) / 80)

    return 0.6 * kw_score + 0.4 * length_score


def _tag_score(action: SupportAction, ticket: Dict[str, Any]) -> float:
    """
    Partial credit: fraction of gt_tags that the agent included.
    Missing all = 0.0, having all = 1.0.
    Extra tags don't hurt.
    """
    gt_tags: List[str] = [t.lower() for t in ticket.get("gt_tags", [])]
    if not gt_tags:
        return 1.0
    agent_tags = [t.lower() for t in (action.tags or [])]
    hits = sum(1 for gt in gt_tags if any(gt in a or a in gt for a in agent_tags))
    return hits / len(gt_tags)


def _penalty(action: SupportAction) -> float:
    """
    Deductions:
      - Empty response: -0.3
      - Invalid category: -0.1
      - Invalid priority: -0.1
    """
    pen = 0.0
    if not (action.response or "").strip():
        pen += 0.3
    if (action.category or "").lower() not in VALID_CATEGORIES:
        pen += 0.1
    if (action.priority or "").lower() not in VALID_PRIORITIES:
        pen += 0.1
    return pen


# ---------------------------------------------------------------------------
# Per-difficulty graders
# ---------------------------------------------------------------------------

def grade_easy(action: SupportAction, ticket: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Easy grader: category(40%) priority(30%) response(20%) tags(10%)."""
    c = _category_score(action, ticket)
    p = _priority_score(action, ticket)
    r = _response_score(action, ticket)
    t = _tag_score(action, ticket)
    pen = _penalty(action)

    raw = 0.40 * c + 0.30 * p + 0.20 * r + 0.10 * t
    score = max(0.0, min(1.0, raw - pen))
    return score, {
        "category": c, "priority": p, "response": r, "tags": t, "penalty": -pen
    }


def grade_medium(action: SupportAction, ticket: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Medium grader: category(25%) priority(25%) escalation(25%) response(20%) tags(5%)."""
    c = _category_score(action, ticket)
    p = _priority_score(action, ticket)
    e = _escalation_score(action, ticket)
    r = _response_score(action, ticket)
    t = _tag_score(action, ticket)
    pen = _penalty(action)

    raw = 0.25 * c + 0.25 * p + 0.25 * e + 0.20 * r + 0.05 * t
    score = max(0.0, min(1.0, raw - pen))
    return score, {
        "category": c, "priority": p, "escalation": e, "response": r, "tags": t, "penalty": -pen
    }


def grade_hard(action: SupportAction, ticket: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Hard grader: category(15%) priority(20%) escalation(30%) response(25%) tags(10%)."""
    c = _category_score(action, ticket)
    p = _priority_score(action, ticket)
    e = _escalation_score(action, ticket)
    r = _response_score(action, ticket)
    t = _tag_score(action, ticket)
    pen = _penalty(action)

    raw = 0.15 * c + 0.20 * p + 0.30 * e + 0.25 * r + 0.10 * t
    score = max(0.0, min(1.0, raw - pen))
    return score, {
        "category": c, "priority": p, "escalation": e, "response": r, "tags": t, "penalty": -pen
    }


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade(task: str, action: SupportAction, ticket: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Grade an action against a ticket for the given task difficulty."""
    grader = GRADERS.get(task, grade_medium)
    return grader(action, ticket)
