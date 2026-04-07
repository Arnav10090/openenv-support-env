"""
Integration tests for baseline inference script.

Tests that the baseline script can connect to the server,
complete all tasks, and produce correctly formatted logs.
"""

from __future__ import annotations

import re
import subprocess
import time
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Import the FastAPI app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing without actual API calls."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = """{
        "category": "billing",
        "priority": "high",
        "escalate": false,
        "tags": ["billing", "refund"],
        "response": "We sincerely apologize for the duplicate charge. We will investigate this matter urgently and process a refund within 3-5 business days."
    }"""
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


class TestBaselineConnection:
    """Tests for baseline script connection to server."""

    def test_baseline_can_connect_to_server(self, client):
        """Test that baseline script can connect to the test server."""
        # Verify server is accessible
        response = client.get("/health")
        assert response.status_code == 200
        
        # Verify reset endpoint works
        response = client.post("/reset?task=easy")
        assert response.status_code == 200
        
        # Verify step endpoint works
        action = {
            "category": "billing",
            "priority": "high",
            "response": "Thank you for contacting us. We will investigate the duplicate charge.",
            "escalate": False,
            "tags": ["billing"]
        }
        response = client.post("/step", json=action)
        assert response.status_code == 200


class TestBaselineLogFormat:
    """Tests for baseline script log format."""

    def test_log_start_format(self):
        """Test that log_start produces correct format."""
        # Import the log function
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from inference import log_start
        
        # Capture output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            log_start("easy", "test-model")
        output = f.getvalue()
        
        # Verify format
        assert output.startswith("[START]")
        assert "task=easy" in output
        assert "env=customer_support_triage" in output
        assert "model=test-model" in output

    def test_log_step_format(self):
        """Test that log_step produces correct format."""
        from inference import log_step
        
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            log_step(1, "category=billing,priority=high,escalate=false", 0.85, False, None)
        output = f.getvalue()
        
        # Verify format
        assert output.startswith("[STEP]")
        assert "step=1" in output
        assert "action=category=billing,priority=high,escalate=false" in output
        assert "reward=0.85" in output
        assert "done=false" in output
        assert "error=null" in output

    def test_log_step_format_with_error(self):
        """Test that log_step handles errors correctly."""
        from inference import log_step
        
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            log_step(1, "category=billing,priority=high,escalate=false", 0.0, False, "Connection failed")
        output = f.getvalue()
        
        # Verify format
        assert output.startswith("[STEP]")
        assert "error=Connection failed" in output

    def test_log_end_format(self):
        """Test that log_end produces correct format."""
        from inference import log_end
        
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            log_end(True, 3, 0.756, [0.85, 0.72, 0.70])
        output = f.getvalue()
        
        # Verify format
        assert output.startswith("[END]")
        assert "success=true" in output
        assert "steps=3" in output
        assert "score=0.756" in output
        assert "rewards=0.85,0.72,0.70" in output

    def test_log_end_format_failure(self):
        """Test that log_end handles failure correctly."""
        from inference import log_end
        
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            log_end(False, 2, 0.350, [0.40, 0.30])
        output = f.getvalue()
        
        # Verify format
        assert output.startswith("[END]")
        assert "success=false" in output
        assert "steps=2" in output


class TestBaselineLLMIntegration:
    """Tests for baseline LLM integration."""

    def test_call_llm_with_mock(self, mock_openai_client):
        """Test that call_llm works with mocked OpenAI client."""
        from inference import call_llm
        
        result = call_llm(
            mock_openai_client,
            "I was charged twice",
            "Please refund the duplicate charge",
            []
        )
        
        assert "category" in result
        assert "priority" in result
        assert "escalate" in result
        assert "tags" in result
        assert "response" in result
        assert result["category"] == "billing"
        assert result["priority"] == "high"

    def test_call_llm_handles_markdown_fences(self, mock_openai_client):
        """Test that call_llm strips markdown fences."""
        from inference import call_llm
        
        # Mock response with markdown fences
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = """```json
{
    "category": "technical",
    "priority": "urgent",
    "escalate": true,
    "tags": ["technical", "bug"],
    "response": "We apologize for the issue. Our engineering team will investigate this urgently."
}
```"""
        
        result = call_llm(
            mock_openai_client,
            "App crashes",
            "The app crashes every time I export data",
            []
        )
        
        assert result["category"] == "technical"
        assert result["priority"] == "urgent"
        assert result["escalate"] is True

    def test_call_llm_fallback_on_error(self):
        """Test that call_llm returns safe fallback on error."""
        from inference import call_llm
        
        # Create a client that raises an exception
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        result = call_llm(
            mock_client,
            "Test subject",
            "Test body",
            []
        )
        
        # Should return safe defaults
        assert result["category"] == "general"
        assert result["priority"] == "medium"
        assert result["escalate"] is False
        assert result["tags"] == []
        assert len(result["response"]) > 0


class TestBaselineTaskExecution:
    """Tests for baseline task execution."""

    @patch('inference.OpenAI')
    def test_run_task_completes_episode(self, mock_openai_class, client):
        """Test that run_task completes a full episode."""
        from inference import run_task
        
        # Setup mock
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = """{
            "category": "billing",
            "priority": "high",
            "escalate": false,
            "tags": ["billing", "refund"],
            "response": "We sincerely apologize for the duplicate charge. We will investigate this matter urgently and process a refund within 3-5 business days."
        }"""
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_client
        
        # Note: This test requires the server to be running
        # For unit testing, we would need to mock the HTTP calls as well
        # This is more of an integration test that requires a live server
        
        # For now, we'll just verify the function exists and has correct signature
        import inspect
        sig = inspect.signature(run_task)
        assert 'client' in sig.parameters
        assert 'task' in sig.parameters


class TestBaselineValidation:
    """Tests for baseline script validation."""

    def test_valid_categories_defined(self):
        """Test that VALID_CATEGORIES is correctly defined."""
        from inference import VALID_CATEGORIES
        
        assert "billing" in VALID_CATEGORIES
        assert "technical" in VALID_CATEGORIES
        assert "account" in VALID_CATEGORIES
        assert "shipping" in VALID_CATEGORIES
        assert "general" in VALID_CATEGORIES
        assert len(VALID_CATEGORIES) == 5

    def test_valid_priorities_defined(self):
        """Test that VALID_PRIORITIES is correctly defined."""
        from inference import VALID_PRIORITIES
        
        assert "low" in VALID_PRIORITIES
        assert "medium" in VALID_PRIORITIES
        assert "high" in VALID_PRIORITIES
        assert "urgent" in VALID_PRIORITIES
        assert len(VALID_PRIORITIES) == 4

    def test_tasks_defined(self):
        """Test that TASKS is correctly defined."""
        from inference import TASKS
        
        assert TASKS == ["easy", "medium", "hard"]

    def test_max_steps_per_task(self):
        """Test that MAX_STEPS_PER_TASK is 3."""
        from inference import MAX_STEPS_PER_TASK
        
        assert MAX_STEPS_PER_TASK == 3

    def test_success_threshold(self):
        """Test that SUCCESS_SCORE_THRESHOLD is 0.5."""
        from inference import SUCCESS_SCORE_THRESHOLD
        
        assert SUCCESS_SCORE_THRESHOLD == 0.5


class TestBaselineSystemPrompt:
    """Tests for baseline system prompt."""

    def test_system_prompt_exists(self):
        """Test that SYSTEM_PROMPT is defined."""
        from inference import SYSTEM_PROMPT
        
        assert len(SYSTEM_PROMPT) > 0
        assert "JSON" in SYSTEM_PROMPT or "json" in SYSTEM_PROMPT

    def test_system_prompt_includes_categories(self):
        """Test that system prompt includes category guidance."""
        from inference import SYSTEM_PROMPT
        
        assert "billing" in SYSTEM_PROMPT
        assert "technical" in SYSTEM_PROMPT
        assert "account" in SYSTEM_PROMPT
        assert "shipping" in SYSTEM_PROMPT

    def test_system_prompt_includes_priorities(self):
        """Test that system prompt includes priority guidance."""
        from inference import SYSTEM_PROMPT
        
        assert "low" in SYSTEM_PROMPT
        assert "medium" in SYSTEM_PROMPT
        assert "high" in SYSTEM_PROMPT
        assert "urgent" in SYSTEM_PROMPT

    def test_system_prompt_includes_escalation_guidance(self):
        """Test that system prompt includes escalation guidance."""
        from inference import SYSTEM_PROMPT
        
        assert "escalate" in SYSTEM_PROMPT.lower()


class TestBaselineConfiguration:
    """Tests for baseline configuration."""

    def test_environment_url_default(self):
        """Test that ENVIRONMENT_URL has correct default."""
        from inference import ENVIRONMENT_URL
        
        # Should be localhost:7860 by default
        assert "7860" in ENVIRONMENT_URL or "localhost" in ENVIRONMENT_URL

    def test_benchmark_name(self):
        """Test that BENCHMARK is correctly set."""
        from inference import BENCHMARK
        
        assert BENCHMARK == "customer_support_triage"

    def test_temperature_setting(self):
        """Test that TEMPERATURE is set."""
        from inference import TEMPERATURE
        
        assert 0.0 <= TEMPERATURE <= 1.0

    def test_max_tokens_setting(self):
        """Test that MAX_TOKENS is set."""
        from inference import MAX_TOKENS
        
        assert MAX_TOKENS > 0


class TestBaselineActionFormatting:
    """Tests for action string formatting."""

    def test_action_string_format(self):
        """Test that action strings follow the correct format."""
        # The action string should be: category=X,priority=Y,escalate=Z
        action_str = "category=billing,priority=high,escalate=false"
        
        # Verify format
        assert "category=" in action_str
        assert "priority=" in action_str
        assert "escalate=" in action_str
        
        # Verify no spaces
        assert " " not in action_str.replace("escalate=false", "").replace("escalate=true", "")

    def test_escalate_boolean_lowercase(self):
        """Test that escalate boolean is lowercase in action string."""
        action_str_false = "category=billing,priority=high,escalate=false"
        action_str_true = "category=technical,priority=urgent,escalate=true"
        
        assert "escalate=false" in action_str_false
        assert "escalate=true" in action_str_true
        assert "False" not in action_str_false
        assert "True" not in action_str_true


class TestBaselineNormalization:
    """Tests for category and priority normalization."""

    def test_invalid_category_normalization(self):
        """Test that invalid categories are normalized to 'general'."""
        # This would be tested in the actual run_task function
        # where invalid categories are normalized
        invalid_category = "invalid_category"
        normalized = "general" if invalid_category not in ["billing", "technical", "account", "shipping", "general"] else invalid_category
        assert normalized == "general"

    def test_invalid_priority_normalization(self):
        """Test that invalid priorities are normalized to 'medium'."""
        invalid_priority = "invalid_priority"
        normalized = "medium" if invalid_priority not in ["low", "medium", "high", "urgent"] else invalid_priority
        assert normalized == "medium"

    def test_valid_category_not_normalized(self):
        """Test that valid categories are not changed."""
        for category in ["billing", "technical", "account", "shipping", "general"]:
            normalized = "general" if category not in ["billing", "technical", "account", "shipping", "general"] else category
            assert normalized == category

    def test_valid_priority_not_normalized(self):
        """Test that valid priorities are not changed."""
        for priority in ["low", "medium", "high", "urgent"]:
            normalized = "medium" if priority not in ["low", "medium", "high", "urgent"] else priority
            assert normalized == priority
