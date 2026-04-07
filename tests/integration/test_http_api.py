"""
Integration tests for HTTP API endpoints.

Tests the FastAPI server endpoints with actual HTTP requests.
Requires the server to be running or uses TestClient for in-process testing.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Reset global state before each test
    import server.app as app_module
    app_module._env = None
    app_module._current_task = app_module.DEFAULT_TASK
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_correct_structure(self, client):
        """Test that /health returns the expected structure."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "task" in data
        assert "version" in data
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["task"] in ("easy", "medium", "hard")

    def test_health_returns_current_task(self, client):
        """Test that /health reflects the current task after reset."""
        # Reset to medium
        client.post("/reset?task=medium")
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["task"] == "medium"


class TestResetEndpoint:
    """Tests for POST /reset endpoint."""

    def test_reset_with_default_task(self, client):
        """Test /reset without task parameter uses default."""
        response = client.post("/reset")
        assert response.status_code == 200
        data = response.json()
        
        # Check observation structure
        assert "ticket_id" in data
        assert "subject" in data
        assert "body" in data
        assert "customer_history" in data
        assert "available_categories" in data
        assert "available_priorities" in data
        assert "step_feedback" in data
        assert "reward" in data
        assert "done" in data
        assert "metadata" in data
        
        # Check initial values
        assert data["ticket_id"] != ""
        assert data["subject"] != ""
        assert data["body"] != ""
        assert data["done"] is False
        assert data["reward"] == 0.0
        assert "Episode started" in data["step_feedback"]

    def test_reset_with_easy_task(self, client):
        """Test /reset with task=easy."""
        response = client.post("/reset?task=easy")
        assert response.status_code == 200
        data = response.json()
        assert data["ticket_id"].startswith("E")
        assert data["metadata"]["difficulty"] == "easy"

    def test_reset_with_medium_task(self, client):
        """Test /reset with task=medium."""
        response = client.post("/reset?task=medium")
        assert response.status_code == 200
        data = response.json()
        assert data["ticket_id"].startswith("M")
        assert data["metadata"]["difficulty"] == "medium"

    def test_reset_with_hard_task(self, client):
        """Test /reset with task=hard."""
        response = client.post("/reset?task=hard")
        assert response.status_code == 200
        data = response.json()
        assert data["ticket_id"].startswith("H")
        assert data["metadata"]["difficulty"] == "hard"

    def test_reset_with_invalid_task_returns_400(self, client):
        """Test /reset with invalid task returns 400 error."""
        response = client.post("/reset?task=invalid")
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "invalid" in data["detail"].lower()

    def test_reset_creates_new_episode(self, client):
        """Test that each /reset creates a new episode with different episode_id."""
        # First reset
        client.post("/reset?task=easy")
        response1 = client.get("/state")
        episode_id_1 = response1.json()["episode_id"]
        
        # Second reset
        client.post("/reset?task=easy")
        response2 = client.get("/state")
        episode_id_2 = response2.json()["episode_id"]
        
        # Episode IDs should be different
        assert episode_id_1 != episode_id_2


class TestStepEndpoint:
    """Tests for POST /step endpoint."""

    def test_step_before_reset_returns_400(self, client):
        """Test that /step before /reset returns 400 error."""
        action = {
            "category": "billing",
            "priority": "high",
            "response": "Thank you for contacting us. We will process your refund within 3-5 business days.",
            "escalate": False,
            "tags": ["billing", "refund"]
        }
        response = client.post("/step", json=action)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "reset" in data["detail"].lower()

    def test_step_with_valid_action(self, client):
        """Test /step with valid action returns correct structure."""
        # Reset first
        client.post("/reset?task=easy")
        
        # Submit action
        action = {
            "category": "billing",
            "priority": "high",
            "response": "Thank you for contacting us. We will process your refund within 3-5 business days.",
            "escalate": False,
            "tags": ["billing", "refund"]
        }
        response = client.post("/step", json=action)
        assert response.status_code == 200
        data = response.json()
        
        # Check step response structure
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data
        
        # Check observation structure
        obs = data["observation"]
        assert "ticket_id" in obs
        assert "subject" in obs
        assert "body" in obs
        assert "step_feedback" in obs
        assert "reward" in obs
        assert "done" in obs
        
        # Check reward is in valid range
        assert 0.0 <= data["reward"] <= 1.0
        
        # Check info contains breakdown
        assert "breakdown" in data["info"]
        assert "ticket_id" in data["info"]

    def test_step_with_minimal_action(self, client):
        """Test /step with minimal action (defaults)."""
        client.post("/reset?task=easy")
        
        action = {}  # Use all defaults
        response = client.post("/step", json=action)
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["reward"] <= 1.0

    def test_step_with_invalid_category(self, client):
        """Test /step with invalid category still processes (penalty applied)."""
        client.post("/reset?task=easy")
        
        action = {
            "category": "invalid_category",
            "priority": "high",
            "response": "Thank you for your message. We will look into this matter promptly.",
            "escalate": False,
            "tags": []
        }
        response = client.post("/step", json=action)
        assert response.status_code == 200
        data = response.json()
        # Should still return valid response with penalty
        assert "breakdown" in data["info"]
        assert data["info"]["breakdown"]["penalty"] < 0

    def test_step_with_empty_response(self, client):
        """Test /step with empty response applies penalty."""
        client.post("/reset?task=easy")
        
        action = {
            "category": "billing",
            "priority": "high",
            "response": "",  # Empty response
            "escalate": False,
            "tags": []
        }
        response = client.post("/step", json=action)
        assert response.status_code == 200
        data = response.json()
        # Should have penalty for empty response
        assert data["info"]["breakdown"]["penalty"] < 0


class TestStateEndpoint:
    """Tests for GET /state endpoint."""

    def test_state_before_reset_returns_400(self, client):
        """Test that /state before /reset returns 400 error."""
        response = client.get("/state")
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "reset" in data["detail"].lower()

    def test_state_returns_correct_structure(self, client):
        """Test that /state returns the expected structure."""
        client.post("/reset?task=easy")
        
        response = client.get("/state")
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields
        assert "episode_id" in data
        assert "task_name" in data
        assert "step_count" in data
        assert "total_reward" in data
        assert "tickets_processed" in data
        assert "correct_categories" in data
        assert "correct_priorities" in data
        assert "escalations_made" in data
        assert "escalations_needed" in data
        assert "metadata" in data
        
        # Check initial values
        assert data["task_name"] == "easy"
        assert data["step_count"] == 0
        assert data["total_reward"] == 0.0
        assert data["tickets_processed"] == 0
        assert data["correct_categories"] == 0
        assert data["correct_priorities"] == 0
        assert data["escalations_made"] == 0

    def test_state_updates_after_step(self, client):
        """Test that /state reflects updates after step."""
        client.post("/reset?task=easy")
        
        # Take a step
        action = {
            "category": "billing",
            "priority": "high",
            "response": "Thank you for contacting us. We will process your refund within 3-5 business days.",
            "escalate": False,
            "tags": ["billing", "refund"]
        }
        step_response = client.post("/step", json=action)
        reward = step_response.json()["reward"]
        
        # Check state
        state_response = client.get("/state")
        data = state_response.json()
        
        assert data["step_count"] == 1
        assert data["tickets_processed"] == 1
        assert data["total_reward"] == reward


class TestErrorResponses:
    """Tests for error handling and 400 status codes."""

    def test_invalid_endpoint_returns_404(self, client):
        """Test that invalid endpoint returns 404."""
        response = client.get("/invalid_endpoint")
        assert response.status_code == 404

    def test_reset_with_uppercase_task(self, client):
        """Test that /reset handles case-insensitive task names."""
        response = client.post("/reset?task=EASY")
        assert response.status_code == 200
        data = response.json()
        assert data["ticket_id"].startswith("E")

    def test_reset_with_whitespace_task(self, client):
        """Test that /reset handles task names with whitespace."""
        response = client.post("/reset?task= easy ")
        assert response.status_code == 200
        data = response.json()
        assert data["ticket_id"].startswith("E")


class TestRootEndpoint:
    """Tests for GET / root endpoint."""

    def test_root_returns_service_info(self, client):
        """Test that root endpoint returns service information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["version"] == "1.0.0"
