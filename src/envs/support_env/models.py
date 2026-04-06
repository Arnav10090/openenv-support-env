"""
Customer Support Triage Environment — Type-safe models.

Action:   The agent sends a structured triage decision.
Observation: The agent receives a ticket + context + feedback.
State:    Metadata about the current episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass
class SupportAction:
    """
    The agent's triage decision for a support ticket.

    Fields
    ------
    category : str
        One of: "billing", "technical", "account", "shipping", "general"
    priority : str
        One of: "low", "medium", "high", "urgent"
    response : str
        The draft reply to send to the customer.
    escalate : bool
        Whether to escalate to a human agent.
    tags : List[str]
        Optional tags to attach to the ticket (e.g. ["refund", "angry-customer"]).
    metadata : Dict[str, Any]
        Any extra fields the agent wants to include.
    """

    category: str = ""
    priority: str = "medium"
    response: str = ""
    escalate: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class SupportObservation:
    """
    What the agent observes at each step.

    Fields
    ------
    ticket_id : str
    subject : str
    body : str
    customer_history : List[Dict]   – previous tickets from same customer
    available_categories : List[str]
    available_priorities : List[str]
    step_feedback : str             – textual feedback from last action
    reward : float                  – reward received for last action
    done : bool
    metadata : Dict[str, Any]
    """

    ticket_id: str = ""
    subject: str = ""
    body: str = ""
    customer_history: List[Dict[str, Any]] = field(default_factory=list)
    available_categories: List[str] = field(default_factory=lambda: [
        "billing", "technical", "account", "shipping", "general"
    ])
    available_priorities: List[str] = field(default_factory=lambda: [
        "low", "medium", "high", "urgent"
    ])
    step_feedback: str = ""
    reward: float = 0.0
    done: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class SupportState:
    """
    Full episode state (metadata for the grader).

    Fields
    ------
    episode_id : str
    task_name : str          – "easy", "medium", or "hard"
    step_count : int
    total_reward : float
    tickets_processed : int
    correct_categories : int
    correct_priorities : int
    escalations_made : int
    escalations_needed : int
    """

    episode_id: str = ""
    task_name: str = "easy"
    step_count: int = 0
    total_reward: float = 0.0
    tickets_processed: int = 0
    correct_categories: int = 0
    correct_priorities: int = 0
    escalations_made: int = 0
    escalations_needed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
