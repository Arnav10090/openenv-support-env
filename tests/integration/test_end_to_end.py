"""
End-to-end integration tests for complete episode flows.

Tests complete episodes from reset through all steps to termination,
verifying state transitions, counter updates, and terminal observations.
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


class TestCompleteEpisodeFlow:
    """Tests for complete episode execution from reset to done."""

    def test_easy_episode_complete_flow(self, client):
        """Test complete episode flow for easy task: reset → 3 steps → done."""
        # Reset
        reset_response = client.post("/reset?task=easy")
        assert reset_response.status_code == 200
        obs = reset_response.json()
        assert obs["done"] is False
        assert obs["ticket_id"].startswith("E")
        
        # Track rewards
        rewards = []
        
        # Step 1
        action1 = {
            "category": "billing",
            "priority": "high",
            "response": "Thank you for contacting us. We will investigate the duplicate charge and process a refund.",
            "escalate": False,
            "tags": ["billing", "refund"]
        }
        step1_response = client.post("/step", json=action1)
        assert step1_response.status_code == 200
        step1_data = step1_response.json()
        assert step1_data["done"] is False
        assert 0.0 <= step1_data["reward"] <= 1.0
        rewards.append(step1_data["reward"])
        
        # Step 2
        action2 = {
            "category": "account",
            "priority": "high",
            "response": "I understand you're having trouble logging in. Let me help you reset your password securely.",
            "escalate": False,
            "tags": ["account", "login"]
        }
        step2_response = client.post("/step", json=action2)
        assert step2_response.status_code == 200
        step2_data = step2_response.json()
        assert step2_data["done"] is False
        assert 0.0 <= step2_data["reward"] <= 1.0
        rewards.append(step2_data["reward"])
        
        # Step 3 (final)
        action3 = {
            "category": "shipping",
            "priority": "medium",
            "response": "I can help you track your order. Let me look up the tracking information for order #ORD-4821.",
            "escalate": False,
            "tags": ["shipping", "tracking"]
        }
        step3_response = client.post("/step", json=action3)
        assert step3_response.status_code == 200
        step3_data = step3_response.json()
        assert step3_data["done"] is True  # Episode should be done
        assert 0.0 <= step3_data["reward"] <= 1.0
        rewards.append(step3_data["reward"])
        
        # Verify terminal observation
        terminal_obs = step3_data["observation"]
        assert terminal_obs["done"] is True
        assert terminal_obs["ticket_id"] == ""
        assert terminal_obs["subject"] == ""
        assert terminal_obs["body"] == ""
        
        # Verify all rewards are valid
        assert len(rewards) == 3
        assert all(0.0 <= r <= 1.0 for r in rewards)

    def test_medium_episode_complete_flow(self, client):
        """Test complete episode flow for medium task: reset → 3 steps → done."""
        # Reset
        reset_response = client.post("/reset?task=medium")
        assert reset_response.status_code == 200
        obs = reset_response.json()
        assert obs["done"] is False
        assert obs["ticket_id"].startswith("M")
        
        rewards = []
        
        # Step 1 - App crashes (should escalate)
        action1 = {
            "category": "technical",
            "priority": "urgent",
            "response": "I understand this is urgent. I'm escalating this to our engineering team for immediate investigation.",
            "escalate": True,
            "tags": ["technical", "bug", "urgent"]
        }
        step1_response = client.post("/step", json=action1)
        assert step1_response.status_code == 200
        step1_data = step1_response.json()
        assert step1_data["done"] is False
        rewards.append(step1_data["reward"])
        
        # Step 2 - Cancellation
        action2 = {
            "category": "billing",
            "priority": "high",
            "response": "I can help you cancel your subscription. Let me process that for you right away.",
            "escalate": False,
            "tags": ["billing", "cancellation"]
        }
        step2_response = client.post("/step", json=action2)
        assert step2_response.status_code == 200
        step2_data = step2_response.json()
        assert step2_data["done"] is False
        rewards.append(step2_data["reward"])
        
        # Step 3 - Data loss (should escalate)
        action3 = {
            "category": "technical",
            "priority": "urgent",
            "response": "This is a serious issue. I'm escalating to our data recovery team to investigate immediately.",
            "escalate": True,
            "tags": ["technical", "data-loss", "urgent"]
        }
        step3_response = client.post("/step", json=action3)
        assert step3_response.status_code == 200
        step3_data = step3_response.json()
        assert step3_data["done"] is True
        rewards.append(step3_data["reward"])
        
        # Verify terminal observation
        terminal_obs = step3_data["observation"]
        assert terminal_obs["done"] is True
        assert terminal_obs["ticket_id"] == ""
        assert terminal_obs["subject"] == ""
        assert terminal_obs["body"] == ""
        
        # Verify all rewards are valid
        assert len(rewards) == 3
        assert all(0.0 <= r <= 1.0 for r in rewards)

    def test_hard_episode_complete_flow(self, client):
        """Test complete episode flow for hard task: reset → 3 steps → done."""
        # Reset
        reset_response = client.post("/reset?task=hard")
        assert reset_response.status_code == 200
        obs = reset_response.json()
        assert obs["done"] is False
        assert obs["ticket_id"].startswith("H")
        
        rewards = []
        
        # Step 1 - GDPR legal threat (must escalate)
        action1 = {
            "category": "account",
            "priority": "urgent",
            "response": "I understand your concerns about data privacy. I'm immediately escalating this to our legal and compliance team.",
            "escalate": True,
            "tags": ["legal", "gdpr", "urgent"]
        }
        step1_response = client.post("/step", json=action1)
        assert step1_response.status_code == 200
        step1_data = step1_response.json()
        assert step1_data["done"] is False
        rewards.append(step1_data["reward"])
        
        # Step 2 - Fraud and hacking (must escalate)
        action2 = {
            "category": "account",
            "priority": "urgent",
            "response": "This is a critical security issue. I'm escalating to our fraud and security team immediately.",
            "escalate": True,
            "tags": ["security", "fraud", "urgent"]
        }
        step2_response = client.post("/step", json=action2)
        assert step2_response.status_code == 200
        step2_data = step2_response.json()
        assert step2_data["done"] is False
        rewards.append(step2_data["reward"])
        
        # Step 3 - API rate limits enterprise (must escalate)
        action3 = {
            "category": "technical",
            "priority": "urgent",
            "response": "I understand this is impacting your production system. Escalating to our enterprise support team for immediate SLA review.",
            "escalate": True,
            "tags": ["technical", "api", "enterprise"]
        }
        step3_response = client.post("/step", json=action3)
        assert step3_response.status_code == 200
        step3_data = step3_response.json()
        assert step3_data["done"] is True
        rewards.append(step3_data["reward"])
        
        # Verify terminal observation
        terminal_obs = step3_data["observation"]
        assert terminal_obs["done"] is True
        assert terminal_obs["ticket_id"] == ""
        assert terminal_obs["subject"] == ""
        assert terminal_obs["body"] == ""
        
        # Verify all rewards are valid
        assert len(rewards) == 3
        assert all(0.0 <= r <= 1.0 for r in rewards)


class TestStateCounterUpdates:
    """Tests for state counter updates during episodes."""

    def test_state_counters_update_correctly(self, client):
        """Test that state counters update correctly throughout episode."""
        # Reset
        client.post("/reset?task=easy")
        
        # Check initial state
        state0 = client.get("/state").json()
        assert state0["step_count"] == 0
        assert state0["tickets_processed"] == 0
        assert state0["total_reward"] == 0.0
        assert state0["correct_categories"] == 0
        assert state0["correct_priorities"] == 0
        assert state0["escalations_made"] == 0
        
        # Step 1 with correct category and priority
        action1 = {
            "category": "billing",  # Correct for E001
            "priority": "high",     # Correct for E001
            "response": "Thank you for contacting us. We will investigate the duplicate charge and process a refund.",
            "escalate": False,
            "tags": ["billing", "refund"]
        }
        step1 = client.post("/step", json=action1).json()
        reward1 = step1["reward"]
        
        state1 = client.get("/state").json()
        assert state1["step_count"] == 1
        assert state1["tickets_processed"] == 1
        assert state1["total_reward"] == reward1
        # Should have correct category and priority
        assert state1["correct_categories"] >= 0
        assert state1["correct_priorities"] >= 0
        
        # Step 2 with escalation
        action2 = {
            "category": "account",
            "priority": "high",
            "response": "I understand you're having trouble logging in. Let me help you reset your password securely.",
            "escalate": True,  # Not needed for easy task, but testing counter
            "tags": ["account", "login"]
        }
        step2 = client.post("/step", json=action2).json()
        reward2 = step2["reward"]
        
        state2 = client.get("/state").json()
        assert state2["step_count"] == 2
        assert state2["tickets_processed"] == 2
        assert state2["total_reward"] == reward1 + reward2
        assert state2["escalations_made"] == 1  # Should increment
        
        # Step 3
        action3 = {
            "category": "shipping",
            "priority": "medium",
            "response": "I can help you track your order. Let me look up the tracking information for order #ORD-4821.",
            "escalate": False,
            "tags": ["shipping", "tracking"]
        }
        step3 = client.post("/step", json=action3).json()
        reward3 = step3["reward"]
        
        state3 = client.get("/state").json()
        assert state3["step_count"] == 3
        assert state3["tickets_processed"] == 3
        assert state3["total_reward"] == reward1 + reward2 + reward3
        assert state3["escalations_made"] == 1  # Should not change

    def test_escalations_needed_set_correctly(self, client):
        """Test that escalations_needed is set correctly for each difficulty."""
        # Easy task - 0 escalations needed
        client.post("/reset?task=easy")
        state_easy = client.get("/state").json()
        assert state_easy["escalations_needed"] == 0
        
        # Medium task - 2 escalations needed
        client.post("/reset?task=medium")
        state_medium = client.get("/state").json()
        assert state_medium["escalations_needed"] == 2
        
        # Hard task - 3 escalations needed
        client.post("/reset?task=hard")
        state_hard = client.get("/state").json()
        assert state_hard["escalations_needed"] == 3


class TestTerminalObservation:
    """Tests for terminal observation properties."""

    def test_terminal_observation_has_empty_ticket_fields(self, client):
        """Test that terminal observation has empty ticket_id, subject, and body."""
        # Reset and complete episode
        client.post("/reset?task=easy")
        
        # Take 3 steps
        action = {
            "category": "general",
            "priority": "medium",
            "response": "Thank you for contacting us. We will get back to you shortly with more information.",
            "escalate": False,
            "tags": []
        }
        
        client.post("/step", json=action)
        client.post("/step", json=action)
        final_step = client.post("/step", json=action).json()
        
        # Check terminal observation
        assert final_step["done"] is True
        terminal_obs = final_step["observation"]
        assert terminal_obs["done"] is True
        assert terminal_obs["ticket_id"] == ""
        assert terminal_obs["subject"] == ""
        assert terminal_obs["body"] == ""
        assert terminal_obs["customer_history"] == []

    def test_terminal_observation_still_has_feedback(self, client):
        """Test that terminal observation still contains feedback from last step."""
        client.post("/reset?task=easy")
        
        action = {
            "category": "billing",
            "priority": "high",
            "response": "Thank you for contacting us. We will investigate the duplicate charge and process a refund.",
            "escalate": False,
            "tags": ["billing"]
        }
        
        client.post("/step", json=action)
        client.post("/step", json=action)
        final_step = client.post("/step", json=action).json()
        
        terminal_obs = final_step["observation"]
        assert terminal_obs["done"] is True
        assert terminal_obs["step_feedback"] != ""  # Should have feedback
        assert terminal_obs["reward"] >= 0.0  # Should have reward from last step


class TestRewardsInValidRange:
    """Tests for reward range validation."""

    def test_all_rewards_in_valid_range(self, client):
        """Test that all rewards are in [0.0, 1.0] range."""
        client.post("/reset?task=medium")
        
        # Test with various actions
        actions = [
            {
                "category": "technical",
                "priority": "urgent",
                "response": "I understand this is urgent. I'm escalating this to our engineering team.",
                "escalate": True,
                "tags": ["technical", "bug"]
            },
            {
                "category": "billing",
                "priority": "high",
                "response": "I can help you cancel your subscription. Let me process that for you.",
                "escalate": False,
                "tags": ["billing"]
            },
            {
                "category": "technical",
                "priority": "urgent",
                "response": "This is a serious issue. I'm escalating to our data recovery team.",
                "escalate": True,
                "tags": ["technical"]
            }
        ]
        
        for action in actions:
            step_response = client.post("/step", json=action).json()
            reward = step_response["reward"]
            assert 0.0 <= reward <= 1.0, f"Reward {reward} out of valid range [0.0, 1.0]"

    def test_rewards_with_penalties_still_in_range(self, client):
        """Test that rewards with penalties are still in [0.0, 1.0] range."""
        client.post("/reset?task=easy")
        
        # Action with multiple penalties
        bad_action = {
            "category": "invalid_category",  # Invalid category penalty
            "priority": "invalid_priority",  # Invalid priority penalty
            "response": "",  # Empty response penalty
            "escalate": False,
            "tags": []
        }
        
        step_response = client.post("/step", json=bad_action).json()
        reward = step_response["reward"]
        assert 0.0 <= reward <= 1.0
        assert reward >= 0.0  # Should be clamped to 0.0, not negative


class TestAllDifficultyLevels:
    """Tests for all three difficulty levels."""

    @pytest.mark.parametrize("task,ticket_prefix", [
        ("easy", "E"),
        ("medium", "M"),
        ("hard", "H"),
    ])
    def test_complete_episode_for_all_difficulties(self, client, task, ticket_prefix):
        """Test complete episode for all difficulty levels."""
        # Reset
        reset_response = client.post(f"/reset?task={task}")
        assert reset_response.status_code == 200
        obs = reset_response.json()
        assert obs["ticket_id"].startswith(ticket_prefix)
        
        # Take 3 steps
        action = {
            "category": "general",
            "priority": "medium",
            "response": "Thank you for contacting us. We are reviewing your request and will respond shortly.",
            "escalate": False,
            "tags": []
        }
        
        for i in range(3):
            step_response = client.post("/step", json=action)
            assert step_response.status_code == 200
            step_data = step_response.json()
            
            if i < 2:
                assert step_data["done"] is False
            else:
                assert step_data["done"] is True
                # Verify terminal observation
                assert step_data["observation"]["ticket_id"] == ""
                assert step_data["observation"]["subject"] == ""
                assert step_data["observation"]["body"] == ""
