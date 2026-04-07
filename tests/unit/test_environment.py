"""
Unit tests for SupportEnvironment class.

Tests verify:
- reset() initializes state correctly
- step() before reset() raises RuntimeError
- step() updates all state counters
- Episode termination sets done=True
- Terminal observation has empty ticket fields
- Feedback string format
- Invalid task raises ValueError
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
import uuid

from envs.support_env.environment import SupportEnvironment
from envs.support_env.models import SupportAction, SupportObservation, SupportState


class TestSupportEnvironmentInit:
    """Test SupportEnvironment initialization."""

    def test_valid_task_easy(self):
        """Test that environment can be initialized with 'easy' task."""
        env = SupportEnvironment(task="easy")
        assert env.task == "easy"
        assert len(env._tickets) == 3
        assert env._current_ticket_idx == 0
        assert env._episode_started is False

    def test_valid_task_medium(self):
        """Test that environment can be initialized with 'medium' task."""
        env = SupportEnvironment(task="medium")
        assert env.task == "medium"
        assert len(env._tickets) == 3
        assert env._current_ticket_idx == 0
        assert env._episode_started is False

    def test_valid_task_hard(self):
        """Test that environment can be initialized with 'hard' task."""
        env = SupportEnvironment(task="hard")
        assert env.task == "hard"
        assert len(env._tickets) == 3
        assert env._current_ticket_idx == 0
        assert env._episode_started is False

    def test_invalid_task_raises_value_error(self):
        """Test that invalid task raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SupportEnvironment(task="invalid")
        
        assert "task must be one of" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_default_task_is_easy(self):
        """Test that default task is 'easy'."""
        env = SupportEnvironment()
        assert env.task == "easy"


class TestSupportEnvironmentReset:
    """Test SupportEnvironment reset() method."""

    def test_reset_initializes_state_correctly(self):
        """Test that reset() initializes state with correct values."""
        env = SupportEnvironment(task="easy")
        obs = env.reset()
        
        # Check state initialization
        state = env.state
        assert state.episode_id != ""  # Should be a UUID
        assert state.task_name == "easy"
        assert state.step_count == 0
        assert state.total_reward == 0.0
        assert state.tickets_processed == 0
        assert state.correct_categories == 0
        assert state.correct_priorities == 0
        assert state.escalations_made == 0
        assert state.escalations_needed == 0  # Easy task has no escalations

    def test_reset_sets_episode_started_flag(self):
        """Test that reset() sets _episode_started to True."""
        env = SupportEnvironment(task="easy")
        assert env._episode_started is False
        
        env.reset()
        assert env._episode_started is True

    def test_reset_resets_ticket_index(self):
        """Test that reset() resets _current_ticket_idx to 0."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        # Advance through some tickets
        action = SupportAction(category="billing", priority="high", response="Test response")
        env.step(action)
        assert env._current_ticket_idx == 1
        
        # Reset should bring it back to 0
        env.reset()
        assert env._current_ticket_idx == 0

    def test_reset_generates_unique_episode_id(self):
        """Test that reset() generates a unique episode_id each time."""
        env = SupportEnvironment(task="easy")
        
        obs1 = env.reset()
        episode_id1 = env.state.episode_id
        
        obs2 = env.reset()
        episode_id2 = env.state.episode_id
        
        assert episode_id1 != episode_id2
        assert episode_id1 != ""
        assert episode_id2 != ""

    def test_reset_returns_first_ticket_observation(self):
        """Test that reset() returns observation with first ticket."""
        env = SupportEnvironment(task="easy")
        obs = env.reset()
        
        assert isinstance(obs, SupportObservation)
        assert obs.ticket_id == "E001"
        assert obs.subject != ""
        assert obs.body != ""
        assert obs.done is False
        assert obs.step_feedback == "Episode started. Good luck!"
        assert obs.reward == 0.0

    def test_reset_computes_escalations_needed_for_easy(self):
        """Test that reset() computes escalations_needed correctly for easy task."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        assert env.state.escalations_needed == 0

    def test_reset_computes_escalations_needed_for_medium(self):
        """Test that reset() computes escalations_needed correctly for medium task."""
        env = SupportEnvironment(task="medium")
        env.reset()
        
        assert env.state.escalations_needed == 2  # M001 and M003

    def test_reset_computes_escalations_needed_for_hard(self):
        """Test that reset() computes escalations_needed correctly for hard task."""
        env = SupportEnvironment(task="hard")
        env.reset()
        
        assert env.state.escalations_needed == 3  # All hard tickets


class TestSupportEnvironmentStep:
    """Test SupportEnvironment step() method."""

    def test_step_before_reset_raises_runtime_error(self):
        """Test that calling step() before reset() raises RuntimeError."""
        env = SupportEnvironment(task="easy")
        action = SupportAction(category="billing", priority="high", response="Test response")
        
        with pytest.raises(RuntimeError) as exc_info:
            env.step(action)
        
        assert "Call reset() before step()" in str(exc_info.value)

    def test_step_returns_correct_tuple_structure(self):
        """Test that step() returns (observation, reward, done, info) tuple."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        result = env.step(action)
        
        assert isinstance(result, tuple)
        assert len(result) == 4
        
        obs, reward, done, info = result
        assert isinstance(obs, SupportObservation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_updates_step_count(self):
        """Test that step() increments step_count."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        assert env.state.step_count == 0
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        env.step(action)
        
        assert env.state.step_count == 1

    def test_step_updates_total_reward(self):
        """Test that step() adds reward to total_reward."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        assert env.state.total_reward == 0.0
        
        action = SupportAction(category="billing", priority="high", response="Test response with sufficient length to meet requirements")
        obs, reward, done, info = env.step(action)
        
        assert env.state.total_reward == reward
        assert env.state.total_reward > 0.0

    def test_step_updates_tickets_processed(self):
        """Test that step() increments tickets_processed."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        assert env.state.tickets_processed == 0
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        env.step(action)
        
        assert env.state.tickets_processed == 1

    def test_step_updates_correct_categories_on_match(self):
        """Test that step() increments correct_categories when category is correct."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        assert env.state.correct_categories == 0
        
        # E001 has gt_category="billing"
        action = SupportAction(category="billing", priority="high", response="Test response")
        env.step(action)
        
        assert env.state.correct_categories == 1

    def test_step_does_not_update_correct_categories_on_mismatch(self):
        """Test that step() does not increment correct_categories when category is wrong."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        assert env.state.correct_categories == 0
        
        # E001 has gt_category="billing", but we submit "technical"
        action = SupportAction(category="technical", priority="high", response="Test response")
        env.step(action)
        
        assert env.state.correct_categories == 0

    def test_step_updates_correct_priorities_on_match(self):
        """Test that step() increments correct_priorities when priority is correct."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        assert env.state.correct_priorities == 0
        
        # E001 has gt_priority="high"
        action = SupportAction(category="billing", priority="high", response="Test response")
        env.step(action)
        
        assert env.state.correct_priorities == 1

    def test_step_does_not_update_correct_priorities_on_mismatch(self):
        """Test that step() does not increment correct_priorities when priority is wrong."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        assert env.state.correct_priorities == 0
        
        # E001 has gt_priority="high", but we submit "low"
        action = SupportAction(category="billing", priority="low", response="Test response")
        env.step(action)
        
        assert env.state.correct_priorities == 0

    def test_step_updates_escalations_made_when_escalate_true(self):
        """Test that step() increments escalations_made when action.escalate is True."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        assert env.state.escalations_made == 0
        
        action = SupportAction(category="billing", priority="high", response="Test response", escalate=True)
        env.step(action)
        
        assert env.state.escalations_made == 1

    def test_step_does_not_update_escalations_made_when_escalate_false(self):
        """Test that step() does not increment escalations_made when action.escalate is False."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        assert env.state.escalations_made == 0
        
        action = SupportAction(category="billing", priority="high", response="Test response", escalate=False)
        env.step(action)
        
        assert env.state.escalations_made == 0

    def test_step_advances_ticket_index(self):
        """Test that step() advances _current_ticket_idx."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        assert env._current_ticket_idx == 0
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        env.step(action)
        
        assert env._current_ticket_idx == 1

    def test_step_returns_info_with_breakdown_and_ticket_id(self):
        """Test that step() returns info dict with breakdown and ticket_id."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        obs, reward, done, info = env.step(action)
        
        assert "breakdown" in info
        assert "ticket_id" in info
        assert isinstance(info["breakdown"], dict)
        assert info["ticket_id"] == "E001"


class TestSupportEnvironmentEpisodeTermination:
    """Test episode termination behavior."""

    def test_episode_terminates_after_all_tickets(self):
        """Test that done=True after processing all tickets."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        
        # Process first two tickets
        obs1, reward1, done1, info1 = env.step(action)
        assert done1 is False
        
        obs2, reward2, done2, info2 = env.step(action)
        assert done2 is False
        
        # Process third (last) ticket
        obs3, reward3, done3, info3 = env.step(action)
        assert done3 is True

    def test_terminal_observation_has_empty_ticket_fields(self):
        """Test that terminal observation has empty ticket_id, subject, and body."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        
        # Process all three tickets
        env.step(action)
        env.step(action)
        obs, reward, done, info = env.step(action)
        
        assert done is True
        assert obs.ticket_id == ""
        assert obs.subject == ""
        assert obs.body == ""
        assert obs.done is True

    def test_terminal_observation_has_feedback_and_reward(self):
        """Test that terminal observation still has feedback and reward from last step."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        
        # Process all three tickets
        env.step(action)
        env.step(action)
        obs, reward, done, info = env.step(action)
        
        assert done is True
        assert obs.step_feedback != ""
        assert obs.reward == reward


class TestSupportEnvironmentFeedback:
    """Test feedback string generation."""

    def test_feedback_contains_reward(self):
        """Test that feedback string contains reward value."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        obs, reward, done, info = env.step(action)
        
        assert "Reward:" in obs.step_feedback
        assert f"{reward:.2f}" in obs.step_feedback

    def test_feedback_contains_category_result(self):
        """Test that feedback string contains category result."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        obs, reward, done, info = env.step(action)
        
        assert "Category:" in obs.step_feedback

    def test_feedback_contains_priority_result(self):
        """Test that feedback string contains priority result."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        obs, reward, done, info = env.step(action)
        
        assert "Priority:" in obs.step_feedback

    def test_feedback_contains_escalation_for_medium_task(self):
        """Test that feedback string contains escalation result for medium task."""
        env = SupportEnvironment(task="medium")
        env.reset()
        
        action = SupportAction(category="technical", priority="urgent", response="Test response", escalate=True)
        obs, reward, done, info = env.step(action)
        
        assert "Escalation:" in obs.step_feedback

    def test_feedback_contains_penalty_when_applied(self):
        """Test that feedback string contains penalty when applied."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        # Submit action with empty response to trigger penalty
        action = SupportAction(category="billing", priority="high", response="")
        obs, reward, done, info = env.step(action)
        
        assert "Penalty applied:" in obs.step_feedback

    def test_feedback_format_uses_pipe_separator(self):
        """Test that feedback string uses pipe separator between components."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        obs, reward, done, info = env.step(action)
        
        assert " | " in obs.step_feedback


class TestSupportEnvironmentStateProperty:
    """Test state property."""

    def test_state_property_returns_support_state(self):
        """Test that state property returns SupportState instance."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        state = env.state
        assert isinstance(state, SupportState)

    def test_state_property_reflects_current_state(self):
        """Test that state property reflects current episode state."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        env.step(action)
        
        state = env.state
        assert state.step_count == 1
        assert state.tickets_processed == 1


class TestSupportEnvironmentMultipleEpisodes:
    """Test multiple episode scenarios."""

    def test_reset_can_be_called_multiple_times(self):
        """Test that reset() can be called multiple times to start new episodes."""
        env = SupportEnvironment(task="easy")
        
        # First episode
        obs1 = env.reset()
        episode_id1 = env.state.episode_id
        
        # Second episode
        obs2 = env.reset()
        episode_id2 = env.state.episode_id
        
        assert episode_id1 != episode_id2
        assert env._current_ticket_idx == 0
        assert env.state.step_count == 0

    def test_reset_after_episode_completion(self):
        """Test that reset() works correctly after completing an episode."""
        env = SupportEnvironment(task="easy")
        env.reset()
        
        action = SupportAction(category="billing", priority="high", response="Test response")
        
        # Complete episode
        env.step(action)
        env.step(action)
        obs, reward, done, info = env.step(action)
        assert done is True
        
        # Reset and start new episode
        obs_new = env.reset()
        assert obs_new.done is False
        assert obs_new.ticket_id == "E001"
        assert env.state.step_count == 0
        assert env._current_ticket_idx == 0
