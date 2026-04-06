"""
Customer Support Triage Environment — Core logic.

An agent receives customer support tickets and must:
  1. Classify the ticket (category, priority)
  2. Decide whether to escalate
  3. Write a draft response
  4. Optionally tag the ticket

Three tasks of increasing difficulty:
  easy   – clear-cut tickets, category & priority are unambiguous
  medium – moderate ambiguity, escalation judgment required
  hard   – complex multi-issue tickets, legal/security/enterprise scenarios
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from .grader import grade
from .models import SupportAction, SupportObservation, SupportState
from .tickets import TASK_TICKETS


class SupportEnvironment:
    """
    Core environment logic (framework-agnostic).
    The FastAPI server wraps this class.
    """

    VALID_TASKS = ("easy", "medium", "hard")

    def __init__(self, task: str = "easy"):
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}, got {task!r}")
        self.task = task
        self._tickets: List[Dict[str, Any]] = list(TASK_TICKETS[task])
        self._state: SupportState = SupportState()
        self._current_ticket_idx: int = 0
        self._episode_started: bool = False

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> SupportObservation:
        """Start a fresh episode and return the first ticket observation."""
        self._current_ticket_idx = 0
        self._episode_started = True
        self._state = SupportState(
            episode_id=str(uuid.uuid4()),
            task_name=self.task,
            step_count=0,
            total_reward=0.0,
            tickets_processed=0,
            correct_categories=0,
            correct_priorities=0,
            escalations_made=0,
            escalations_needed=sum(
                1 for t in self._tickets if t["gt_escalate"]
            ),
        )
        return self._make_observation(feedback="Episode started. Good luck!", reward=0.0, done=False)

    def step(self, action: SupportAction) -> Tuple[SupportObservation, float, bool, Dict[str, Any]]:
        """
        Process agent's triage action.

        Returns
        -------
        observation : SupportObservation
        reward      : float   (partial-progress reward in [0, 1])
        done        : bool
        info        : dict    (grading breakdown)
        """
        if not self._episode_started:
            raise RuntimeError("Call reset() before step().")

        ticket = self._tickets[self._current_ticket_idx]
        reward, breakdown = grade(self.task, action, ticket)

        # Update state counters
        self._state.step_count += 1
        self._state.total_reward += reward
        self._state.tickets_processed += 1

        if breakdown.get("category", 0.0) == 1.0:
            self._state.correct_categories += 1
        if breakdown.get("priority", 0.0) == 1.0:
            self._state.correct_priorities += 1
        if action.escalate:
            self._state.escalations_made += 1

        # Build feedback string
        feedback = self._build_feedback(action, ticket, reward, breakdown)

        # Advance to next ticket
        self._current_ticket_idx += 1
        done = self._current_ticket_idx >= len(self._tickets)

        obs = self._make_observation(feedback=feedback, reward=reward, done=done)
        return obs, reward, done, {"breakdown": breakdown, "ticket_id": ticket["ticket_id"]}

    @property
    def state(self) -> SupportState:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(
        self,
        feedback: str,
        reward: float,
        done: bool,
    ) -> SupportObservation:
        """Build observation from current ticket (or terminal if done)."""
        if done or self._current_ticket_idx >= len(self._tickets):
            # Terminal observation — no more tickets
            return SupportObservation(
                ticket_id="",
                subject="",
                body="",
                customer_history=[],
                step_feedback=feedback,
                reward=reward,
                done=True,
            )

        ticket = self._tickets[self._current_ticket_idx]
        return SupportObservation(
            ticket_id=ticket["ticket_id"],
            subject=ticket["subject"],
            body=ticket["body"],
            customer_history=ticket.get("customer_history", []),
            step_feedback=feedback,
            reward=reward,
            done=False,
            metadata={"difficulty": ticket.get("difficulty", self.task)},
        )

    def _build_feedback(
        self,
        action: SupportAction,
        ticket: Dict[str, Any],
        reward: float,
        breakdown: Dict[str, float],
    ) -> str:
        parts = [f"Reward: {reward:.2f}"]

        cat_ok = breakdown.get("category", 0.0) == 1.0
        pri_ok = breakdown.get("priority", 0.0) >= 0.5
        esc_score = breakdown.get("escalation")

        gt_cat = ticket["gt_category"]
        gt_pri = ticket["gt_priority"]
        parts.append(f"Category: {'✓ correct' if cat_ok else f'✗ expected {gt_cat}'}")
        parts.append(f"Priority: {'✓ correct' if pri_ok else f'✗ expected {gt_pri}'}")
        if esc_score is not None:
            parts.append(
                f"Escalation: {'✓ correct' if esc_score == 1.0 else '✗ wrong decision'}"
            )
        if breakdown.get("penalty", 0.0) < 0:
            parts.append(f"Penalty applied: {breakdown['penalty']:.2f}")

        return " | ".join(parts)
