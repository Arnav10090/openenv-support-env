"""
Unit tests for data models (SupportAction, SupportObservation, SupportState).

Tests verify:
- Field defaults
- Type annotations
- default_factory functions produce correct types
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
from dataclasses import fields
from typing import get_type_hints

from envs.support_env.models import SupportAction, SupportObservation, SupportState


class TestSupportAction:
    """Test SupportAction dataclass field defaults and types."""

    def test_field_defaults(self):
        """Test that SupportAction fields have correct default values."""
        action = SupportAction()
        
        assert action.category == ""
        assert action.priority == "medium"
        assert action.response == ""
        assert action.escalate is False
        assert action.tags == []
        assert action.metadata == {}

    def test_field_type_annotations(self):
        """Test that SupportAction has correct type annotations."""
        hints = get_type_hints(SupportAction)
        
        assert hints['category'] == str
        assert hints['priority'] == str
        assert hints['response'] == str
        assert hints['escalate'] == bool
        # Note: List[str] and Dict[str, Any] are checked as string representations
        assert 'List' in str(hints['tags'])
        assert 'Dict' in str(hints['metadata'])

    def test_default_factory_produces_list(self):
        """Test that tags default_factory produces a list."""
        action1 = SupportAction()
        action2 = SupportAction()
        
        # Verify they are lists
        assert isinstance(action1.tags, list)
        assert isinstance(action2.tags, list)
        
        # Verify they are independent instances
        action1.tags.append("test")
        assert "test" in action1.tags
        assert "test" not in action2.tags

    def test_default_factory_produces_dict(self):
        """Test that metadata default_factory produces a dict."""
        action1 = SupportAction()
        action2 = SupportAction()
        
        # Verify they are dicts
        assert isinstance(action1.metadata, dict)
        assert isinstance(action2.metadata, dict)
        
        # Verify they are independent instances
        action1.metadata["key"] = "value"
        assert "key" in action1.metadata
        assert "key" not in action2.metadata


class TestSupportObservation:
    """Test SupportObservation dataclass field defaults and types."""

    def test_field_defaults(self):
        """Test that SupportObservation fields have correct default values."""
        obs = SupportObservation()
        
        assert obs.ticket_id == ""
        assert obs.subject == ""
        assert obs.body == ""
        assert obs.customer_history == []
        assert obs.available_categories == ["billing", "technical", "account", "shipping", "general"]
        assert obs.available_priorities == ["low", "medium", "high", "urgent"]
        assert obs.step_feedback == ""
        assert obs.reward == 0.0
        assert obs.done is False
        assert obs.metadata == {}

    def test_field_type_annotations(self):
        """Test that SupportObservation has correct type annotations."""
        hints = get_type_hints(SupportObservation)
        
        assert hints['ticket_id'] == str
        assert hints['subject'] == str
        assert hints['body'] == str
        assert hints['step_feedback'] == str
        assert hints['reward'] == float
        assert hints['done'] == bool
        assert 'List' in str(hints['customer_history'])
        assert 'List' in str(hints['available_categories'])
        assert 'List' in str(hints['available_priorities'])
        assert 'Dict' in str(hints['metadata'])

    def test_default_factory_produces_list_for_customer_history(self):
        """Test that customer_history default_factory produces a list."""
        obs1 = SupportObservation()
        obs2 = SupportObservation()
        
        # Verify they are lists
        assert isinstance(obs1.customer_history, list)
        assert isinstance(obs2.customer_history, list)
        
        # Verify they are independent instances
        obs1.customer_history.append({"ticket_id": "T001"})
        assert len(obs1.customer_history) == 1
        assert len(obs2.customer_history) == 0

    def test_default_factory_produces_correct_categories(self):
        """Test that available_categories default_factory produces correct list."""
        obs1 = SupportObservation()
        obs2 = SupportObservation()
        
        expected = ["billing", "technical", "account", "shipping", "general"]
        
        # Verify correct values
        assert obs1.available_categories == expected
        assert obs2.available_categories == expected
        
        # Verify they are independent instances
        obs1.available_categories.append("custom")
        assert "custom" in obs1.available_categories
        assert "custom" not in obs2.available_categories

    def test_default_factory_produces_correct_priorities(self):
        """Test that available_priorities default_factory produces correct list."""
        obs1 = SupportObservation()
        obs2 = SupportObservation()
        
        expected = ["low", "medium", "high", "urgent"]
        
        # Verify correct values
        assert obs1.available_priorities == expected
        assert obs2.available_priorities == expected
        
        # Verify they are independent instances
        obs1.available_priorities.append("critical")
        assert "critical" in obs1.available_priorities
        assert "critical" not in obs2.available_priorities

    def test_default_factory_produces_dict_for_metadata(self):
        """Test that metadata default_factory produces a dict."""
        obs1 = SupportObservation()
        obs2 = SupportObservation()
        
        # Verify they are dicts
        assert isinstance(obs1.metadata, dict)
        assert isinstance(obs2.metadata, dict)
        
        # Verify they are independent instances
        obs1.metadata["key"] = "value"
        assert "key" in obs1.metadata
        assert "key" not in obs2.metadata


class TestSupportState:
    """Test SupportState dataclass field defaults and types."""

    def test_field_defaults(self):
        """Test that SupportState fields have correct default values."""
        state = SupportState()
        
        assert state.episode_id == ""
        assert state.task_name == "easy"
        assert state.step_count == 0
        assert state.total_reward == 0.0
        assert state.tickets_processed == 0
        assert state.correct_categories == 0
        assert state.correct_priorities == 0
        assert state.escalations_made == 0
        assert state.escalations_needed == 0
        assert state.metadata == {}

    def test_field_type_annotations(self):
        """Test that SupportState has correct type annotations."""
        hints = get_type_hints(SupportState)
        
        assert hints['episode_id'] == str
        assert hints['task_name'] == str
        assert hints['step_count'] == int
        assert hints['total_reward'] == float
        assert hints['tickets_processed'] == int
        assert hints['correct_categories'] == int
        assert hints['correct_priorities'] == int
        assert hints['escalations_made'] == int
        assert hints['escalations_needed'] == int
        assert 'Dict' in str(hints['metadata'])

    def test_default_factory_produces_dict(self):
        """Test that metadata default_factory produces a dict."""
        state1 = SupportState()
        state2 = SupportState()
        
        # Verify they are dicts
        assert isinstance(state1.metadata, dict)
        assert isinstance(state2.metadata, dict)
        
        # Verify they are independent instances
        state1.metadata["key"] = "value"
        assert "key" in state1.metadata
        assert "key" not in state2.metadata
