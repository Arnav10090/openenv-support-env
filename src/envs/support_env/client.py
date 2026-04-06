"""
HTTP client for the Customer Support Triage Environment.

Usage (sync):
    from envs.support_env import SupportEnvClient, SupportAction

    client = SupportEnvClient(base_url="http://localhost:8000")
    obs = client.reset()
    while not obs.done:
        action = SupportAction(
            category="billing",
            priority="high",
            response="We apologize for the inconvenience...",
            escalate=False,
        )
        result = client.step(action)
        obs = result["observation"]
        print(f"Reward: {result['reward']:.2f}")
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from .models import SupportAction, SupportObservation, SupportState


class SupportEnvClient:
    """
    Synchronous HTTP client for the Customer Support Triage Environment.
    Connects to a running FastAPI server (local or HF Spaces).
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> SupportObservation:
        """Reset the environment and return the first ticket observation."""
        data = self._post("/reset", {})
        return self._parse_observation(data)

    def step(self, action: SupportAction) -> Dict[str, Any]:
        """
        Submit a triage action.

        Returns dict with keys:
          observation : SupportObservation
          reward      : float
          done        : bool
          info        : dict
        """
        payload = {
            "category": action.category,
            "priority": action.priority,
            "response": action.response,
            "escalate": action.escalate,
            "tags": action.tags,
            "metadata": action.metadata,
        }
        data = self._post("/step", payload)
        return {
            "observation": self._parse_observation(data["observation"]),
            "reward": data["reward"],
            "done": data["done"],
            "info": data.get("info", {}),
        }

    def state(self) -> SupportState:
        """Return current episode state."""
        data = self._get("/state")
        return SupportState(
            episode_id=data.get("episode_id", ""),
            task_name=data.get("task_name", "easy"),
            step_count=data.get("step_count", 0),
            total_reward=data.get("total_reward", 0.0),
            tickets_processed=data.get("tickets_processed", 0),
            correct_categories=data.get("correct_categories", 0),
            correct_priorities=data.get("correct_priorities", 0),
            escalations_made=data.get("escalations_made", 0),
            escalations_needed=data.get("escalations_needed", 0),
        )

    def health(self) -> Dict[str, Any]:
        return self._get("/health")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url + path
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            raise ConnectionError(f"Cannot reach environment at {url}: {e}") from e

    def _get(self, path: str) -> Dict[str, Any]:
        url = self.base_url + path
        req = Request(url, method="GET")
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            raise ConnectionError(f"Cannot reach environment at {url}: {e}") from e

    @staticmethod
    def _parse_observation(data: Dict[str, Any]) -> SupportObservation:
        return SupportObservation(
            ticket_id=data.get("ticket_id", ""),
            subject=data.get("subject", ""),
            body=data.get("body", ""),
            customer_history=data.get("customer_history", []),
            available_categories=data.get(
                "available_categories",
                ["billing", "technical", "account", "shipping", "general"],
            ),
            available_priorities=data.get(
                "available_priorities", ["low", "medium", "high", "urgent"]
            ),
            step_feedback=data.get("step_feedback", ""),
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            metadata=data.get("metadata", {}),
        )
