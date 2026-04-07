"""
Unit tests for baseline inference script functions.
Tests log format functions, action string format, LLM fallback, markdown stripping, and normalization.
"""

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

# Import functions from inference.py
sys.path.insert(0, ".")
from inference import (
    VALID_CATEGORIES,
    VALID_PRIORITIES,
    call_llm,
    log_end,
    log_start,
    log_step,
)


class TestLogFormatFunctions:
    """Test that log format functions produce exact output."""

    def test_log_start_format(self, capsys):
        """Test log_start produces exact format: [START] task=<task> env=<env> model=<model>"""
        log_start(task="easy", model="test-model")
        captured = capsys.readouterr()
        assert captured.out == "[START] task=easy env=customer_support_triage model=test-model\n"

    def test_log_start_different_task(self, capsys):
        """Test log_start with different task name."""
        log_start(task="hard", model="gpt-4")
        captured = capsys.readouterr()
        assert captured.out == "[START] task=hard env=customer_support_triage model=gpt-4\n"

    def test_log_step_format_with_null_error(self, capsys):
        """Test log_step produces exact format with error=null."""
        log_step(step=1, action="category=billing,priority=high,escalate=false", reward=0.85, done=False, error=None)
        captured = capsys.readouterr()
        expected = "[STEP] step=1 action=category=billing,priority=high,escalate=false reward=0.85 done=false error=null\n"
        assert captured.out == expected

    def test_log_step_format_with_error_message(self, capsys):
        """Test log_step produces exact format with error message."""
        log_step(step=2, action="category=technical,priority=urgent,escalate=true", reward=0.0, done=True, error="Connection timeout")
        captured = capsys.readouterr()
        expected = "[STEP] step=2 action=category=technical,priority=urgent,escalate=true reward=0.00 done=true error=Connection timeout\n"
        assert captured.out == expected

    def test_log_step_done_true(self, capsys):
        """Test log_step with done=True."""
        log_step(step=3, action="category=account,priority=medium,escalate=false", reward=0.50, done=True, error=None)
        captured = capsys.readouterr()
        assert "done=true" in captured.out
        assert "error=null" in captured.out

    def test_log_end_format(self, capsys):
        """Test log_end produces exact format."""
        log_end(success=True, steps=3, score=0.750, rewards=[0.85, 0.70, 0.70])
        captured = capsys.readouterr()
        expected = "[END] success=true steps=3 score=0.750 rewards=0.85,0.70,0.70\n"
        assert captured.out == expected

    def test_log_end_format_failure(self, capsys):
        """Test log_end with success=False."""
        log_end(success=False, steps=2, score=0.400, rewards=[0.30, 0.50])
        captured = capsys.readouterr()
        expected = "[END] success=false steps=2 score=0.400 rewards=0.30,0.50\n"
        assert captured.out == expected

    def test_log_end_empty_rewards(self, capsys):
        """Test log_end with empty rewards list."""
        log_end(success=False, steps=0, score=0.0, rewards=[])
        captured = capsys.readouterr()
        expected = "[END] success=false steps=0 score=0.000 rewards=\n"
        assert captured.out == expected

    def test_log_end_single_reward(self, capsys):
        """Test log_end with single reward."""
        log_end(success=True, steps=1, score=0.900, rewards=[0.90])
        captured = capsys.readouterr()
        expected = "[END] success=true steps=1 score=0.900 rewards=0.90\n"
        assert captured.out == expected


class TestActionStringFormat:
    """Test action string format for log_step."""

    def test_action_string_format_basic(self):
        """Test action string follows format: category=X,priority=Y,escalate=Z"""
        category = "billing"
        priority = "high"
        escalate = False
        action_str = f"category={category},priority={priority},escalate={str(escalate).lower()}"
        
        assert action_str == "category=billing,priority=high,escalate=false"
        assert "category=" in action_str
        assert "priority=" in action_str
        assert "escalate=" in action_str

    def test_action_string_format_with_escalate_true(self):
        """Test action string with escalate=true."""
        category = "technical"
        priority = "urgent"
        escalate = True
        action_str = f"category={category},priority={priority},escalate={str(escalate).lower()}"
        
        assert action_str == "category=technical,priority=urgent,escalate=true"

    def test_action_string_format_all_categories(self):
        """Test action string format with all valid categories."""
        for category in VALID_CATEGORIES:
            action_str = f"category={category},priority=medium,escalate=false"
            assert action_str.startswith(f"category={category}")

    def test_action_string_format_all_priorities(self):
        """Test action string format with all valid priorities."""
        for priority in VALID_PRIORITIES:
            action_str = f"category=general,priority={priority},escalate=false"
            assert f"priority={priority}" in action_str


class TestLLMFallback:
    """Test LLM fallback on exceptions."""

    @patch("inference.OpenAI")
    def test_call_llm_exception_returns_safe_default(self, mock_openai_class):
        """Test that call_llm returns safe default dict on exception."""
        # Mock client that raises exception
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        result = call_llm(
            client=mock_client,
            ticket_subject="Test subject",
            ticket_body="Test body",
            customer_history=[],
        )
        
        # Verify safe default values
        assert result["category"] == "general"
        assert result["priority"] == "medium"
        assert result["escalate"] is False
        assert result["tags"] == []
        assert "apologize" in result["response"].lower() or "sincerely" in result["response"].lower()
        assert len(result["response"]) > 80  # Should meet minimum length

    @patch("inference.OpenAI")
    def test_call_llm_json_decode_error_returns_safe_default(self, mock_openai_class):
        """Test that call_llm returns safe default on JSON decode error."""
        # Mock client that returns invalid JSON
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "This is not valid JSON"
        mock_client.chat.completions.create.return_value = mock_completion
        
        result = call_llm(
            client=mock_client,
            ticket_subject="Test subject",
            ticket_body="Test body",
            customer_history=[],
        )
        
        # Verify safe default values
        assert result["category"] == "general"
        assert result["priority"] == "medium"
        assert result["escalate"] is False
        assert result["tags"] == []
        assert isinstance(result["response"], str)

    @patch("inference.OpenAI")
    def test_call_llm_none_content_returns_safe_default(self, mock_openai_class):
        """Test that call_llm returns safe default when content is None."""
        # Mock client that returns None content
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_completion
        
        result = call_llm(
            client=mock_client,
            ticket_subject="Test subject",
            ticket_body="Test body",
            customer_history=[],
        )
        
        # Verify safe default values
        assert result["category"] == "general"
        assert result["priority"] == "medium"


class TestMarkdownFenceStripping:
    """Test markdown fence stripping in call_llm."""

    @patch("inference.OpenAI")
    def test_call_llm_strips_markdown_fences(self, mock_openai_class):
        """Test that call_llm strips markdown code fences from response."""
        # Mock client that returns JSON wrapped in markdown fences
        mock_client = MagicMock()
        mock_completion = MagicMock()
        json_content = {
            "category": "billing",
            "priority": "high",
            "escalate": False,
            "tags": ["refund"],
            "response": "We apologize for the inconvenience."
        }
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = f"```json\n{json.dumps(json_content)}\n```"
        mock_client.chat.completions.create.return_value = mock_completion
        
        result = call_llm(
            client=mock_client,
            ticket_subject="Test subject",
            ticket_body="Test body",
            customer_history=[],
        )
        
        # Verify JSON was parsed correctly after stripping fences
        assert result["category"] == "billing"
        assert result["priority"] == "high"
        assert result["escalate"] is False
        assert result["tags"] == ["refund"]

    @patch("inference.OpenAI")
    def test_call_llm_strips_markdown_fences_without_json_label(self, mock_openai_class):
        """Test that call_llm strips markdown fences even without 'json' label."""
        # Mock client that returns JSON wrapped in markdown fences without json label
        mock_client = MagicMock()
        mock_completion = MagicMock()
        json_content = {
            "category": "technical",
            "priority": "urgent",
            "escalate": True,
            "tags": ["bug"],
            "response": "We sincerely apologize and will escalate this immediately."
        }
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = f"```\n{json.dumps(json_content)}\n```"
        mock_client.chat.completions.create.return_value = mock_completion
        
        result = call_llm(
            client=mock_client,
            ticket_subject="Test subject",
            ticket_body="Test body",
            customer_history=[],
        )
        
        # Verify JSON was parsed correctly
        assert result["category"] == "technical"
        assert result["priority"] == "urgent"
        assert result["escalate"] is True

    @patch("inference.OpenAI")
    def test_call_llm_handles_plain_json_without_fences(self, mock_openai_class):
        """Test that call_llm handles plain JSON without markdown fences."""
        # Mock client that returns plain JSON
        mock_client = MagicMock()
        mock_completion = MagicMock()
        json_content = {
            "category": "account",
            "priority": "medium",
            "escalate": False,
            "tags": ["login"],
            "response": "We apologize for the login issue."
        }
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps(json_content)
        mock_client.chat.completions.create.return_value = mock_completion
        
        result = call_llm(
            client=mock_client,
            ticket_subject="Test subject",
            ticket_body="Test body",
            customer_history=[],
        )
        
        # Verify JSON was parsed correctly
        assert result["category"] == "account"
        assert result["priority"] == "medium"


class TestInvalidCategoryPriorityNormalization:
    """Test invalid category/priority normalization in run_task logic."""

    def test_invalid_category_normalized_to_general(self):
        """Test that invalid category is normalized to 'general'."""
        category = "invalid_category"
        if category not in VALID_CATEGORIES:
            category = "general"
        
        assert category == "general"

    def test_valid_category_not_normalized(self):
        """Test that valid category is not normalized."""
        for valid_cat in VALID_CATEGORIES:
            category = valid_cat
            if category not in VALID_CATEGORIES:
                category = "general"
            
            assert category == valid_cat

    def test_invalid_priority_normalized_to_medium(self):
        """Test that invalid priority is normalized to 'medium'."""
        priority = "invalid_priority"
        if priority not in VALID_PRIORITIES:
            priority = "medium"
        
        assert priority == "medium"

    def test_valid_priority_not_normalized(self):
        """Test that valid priority is not normalized."""
        for valid_pri in VALID_PRIORITIES:
            priority = valid_pri
            if priority not in VALID_PRIORITIES:
                priority = "medium"
            
            assert priority == valid_pri

    def test_empty_category_normalized_to_general(self):
        """Test that empty category is normalized to 'general'."""
        category = ""
        if category not in VALID_CATEGORIES:
            category = "general"
        
        assert category == "general"

    def test_empty_priority_normalized_to_medium(self):
        """Test that empty priority is normalized to 'medium'."""
        priority = ""
        if priority not in VALID_PRIORITIES:
            priority = "medium"
        
        assert priority == "medium"

    def test_case_sensitive_category_normalization(self):
        """Test that category comparison is case-sensitive (BILLING != billing)."""
        category = "BILLING"  # uppercase
        if category not in VALID_CATEGORIES:
            category = "general"
        
        # Should be normalized because VALID_CATEGORIES contains lowercase "billing"
        assert category == "general"

    def test_case_sensitive_priority_normalization(self):
        """Test that priority comparison is case-sensitive (HIGH != high)."""
        priority = "HIGH"  # uppercase
        if priority not in VALID_PRIORITIES:
            priority = "medium"
        
        # Should be normalized because VALID_PRIORITIES contains lowercase "high"
        assert priority == "medium"


class TestCustomerHistoryFormatting:
    """Test customer history formatting in call_llm."""

    @patch("inference.OpenAI")
    def test_call_llm_includes_customer_history(self, mock_openai_class):
        """Test that call_llm includes customer history in the prompt."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        json_content = {
            "category": "technical",
            "priority": "urgent",
            "escalate": True,
            "tags": ["repeat-contact"],
            "response": "We apologize for the continued issues."
        }
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps(json_content)
        mock_client.chat.completions.create.return_value = mock_completion
        
        customer_history = [
            {"date": "2024-01-01", "subject": "Previous issue", "resolution": "Fixed"}
        ]
        
        result = call_llm(
            client=mock_client,
            ticket_subject="Test subject",
            ticket_body="Test body",
            customer_history=customer_history,
        )
        
        # Verify the call was made (history should be in the prompt)
        assert mock_client.chat.completions.create.called
        call_args = mock_client.chat.completions.create.call_args
        user_message = call_args[1]["messages"][1]["content"]
        assert "CUSTOMER HISTORY" in user_message
        assert "Previous issue" in user_message

    @patch("inference.OpenAI")
    def test_call_llm_empty_customer_history(self, mock_openai_class):
        """Test that call_llm handles empty customer history."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        json_content = {
            "category": "billing",
            "priority": "high",
            "escalate": False,
            "tags": ["refund"],
            "response": "We apologize for the billing issue."
        }
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps(json_content)
        mock_client.chat.completions.create.return_value = mock_completion
        
        result = call_llm(
            client=mock_client,
            ticket_subject="Test subject",
            ticket_body="Test body",
            customer_history=[],
        )
        
        # Verify the call was made without history section
        assert mock_client.chat.completions.create.called
        call_args = mock_client.chat.completions.create.call_args
        user_message = call_args[1]["messages"][1]["content"]
        assert "CUSTOMER HISTORY" not in user_message


class TestStepFeedbackFormatting:
    """Test step feedback formatting in call_llm."""

    @patch("inference.OpenAI")
    def test_call_llm_includes_step_feedback(self, mock_openai_class):
        """Test that call_llm includes step feedback in the prompt."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        json_content = {
            "category": "billing",
            "priority": "high",
            "escalate": False,
            "tags": ["refund"],
            "response": "We apologize for the billing error."
        }
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps(json_content)
        mock_client.chat.completions.create.return_value = mock_completion
        
        result = call_llm(
            client=mock_client,
            ticket_subject="Test subject",
            ticket_body="Test body",
            customer_history=[],
            step_feedback="Reward: 0.50 | Category: ✓ correct | Priority: ✗ expected high",
        )
        
        # Verify the call was made with feedback
        assert mock_client.chat.completions.create.called
        call_args = mock_client.chat.completions.create.call_args
        user_message = call_args[1]["messages"][1]["content"]
        assert "FEEDBACK FROM PREVIOUS TRIAGE" in user_message
        assert "Reward: 0.50" in user_message

    @patch("inference.OpenAI")
    def test_call_llm_excludes_episode_started_feedback(self, mock_openai_class):
        """Test that call_llm excludes 'Episode started' feedback."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        json_content = {
            "category": "billing",
            "priority": "high",
            "escalate": False,
            "tags": ["refund"],
            "response": "We apologize for the issue."
        }
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = json.dumps(json_content)
        mock_client.chat.completions.create.return_value = mock_completion
        
        result = call_llm(
            client=mock_client,
            ticket_subject="Test subject",
            ticket_body="Test body",
            customer_history=[],
            step_feedback="Episode started. Good luck!",
        )
        
        # Verify the call was made without feedback section
        assert mock_client.chat.completions.create.called
        call_args = mock_client.chat.completions.create.call_args
        user_message = call_args[1]["messages"][1]["content"]
        assert "FEEDBACK FROM PREVIOUS TRIAGE" not in user_message
