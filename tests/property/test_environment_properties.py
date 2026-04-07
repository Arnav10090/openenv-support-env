"""
Property-based tests for the Customer Support Triage Environment core logic.

These tests use Hypothesis to generate random inputs and verify that the environment
satisfies its correctness properties across a wide range of scenarios.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hypothesis import given, settings, strategies as st

from envs.support_env.environment import SupportEnvironment
from envs.support_env.models import SupportAction
from envs.support_env.tickets import TASK_TICKETS


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

@st.composite
def action_strategy(draw):
    """Generate random SupportAction instances."""
    return SupportAction(
        category=draw(st.sampled_from(["billing", "technical", "account", "shipping", "general", "invalid", ""])),
        priority=draw(st.sampled_from(["low", "medium", "high", "urgent", "invalid", ""])),
        response=draw(st.text(min_size=0, max_size=200)),
        escalate=draw(st.booleans()),
        tags=draw(st.lists(st.text(min_size=1, max_size=15), max_size=5)),
        metadata=draw(st.dictionaries(st.text(min_size=1, max_size=10), st.text(max_size=20), max_size=3))
    )


@st.composite
def task_strategy(draw):
    """Generate random task difficulty."""
    return draw(st.sampled_from(["easy", "medium", "hard"]))


# ---------------------------------------------------------------------------
# Property 10: Step State Transitions
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 10: Step State Transitions
@given(task=task_strategy(), action=action_strategy())
@settings(max_examples=100)
def test_step_state_transitions(task, action):
    """
    Property 10: Step State Transitions
    
    For any valid action on a non-terminal episode, calling step() should 
    increment step_count by 1, add the reward to total_reward, increment 
    tickets_processed by 1, and advance _current_ticket_idx by 1.
    
    **Validates: Requirements 9.3, 9.4, 9.5, 9.6, 9.11**
    """
    # Create environment and reset
    env = SupportEnvironment(task=task)
    env.reset()
    
    # Get the number of tickets for this task
    num_tickets = len(TASK_TICKETS[task])
    
    # Test state transitions for each ticket in the episode
    for ticket_idx in range(num_tickets):
        # Capture state before step
        step_count_before = env.state.step_count
        total_reward_before = env.state.total_reward
        tickets_processed_before = env.state.tickets_processed
        current_ticket_idx_before = env._current_ticket_idx
        
        # Perform step
        obs, reward, done, info = env.step(action)
        
        # Verify state transitions
        assert env.state.step_count == step_count_before + 1, (
            f"step_count should increment by 1. "
            f"Before: {step_count_before}, After: {env.state.step_count}"
        )
        
        assert abs(env.state.total_reward - (total_reward_before + reward)) < 1e-9, (
            f"total_reward should increase by reward amount. "
            f"Before: {total_reward_before:.6f}, Reward: {reward:.6f}, "
            f"After: {env.state.total_reward:.6f}, "
            f"Expected: {total_reward_before + reward:.6f}"
        )
        
        assert env.state.tickets_processed == tickets_processed_before + 1, (
            f"tickets_processed should increment by 1. "
            f"Before: {tickets_processed_before}, After: {env.state.tickets_processed}"
        )
        
        assert env._current_ticket_idx == current_ticket_idx_before + 1, (
            f"_current_ticket_idx should increment by 1. "
            f"Before: {current_ticket_idx_before}, After: {env._current_ticket_idx}"
        )
        
        # Verify done flag is set correctly
        if ticket_idx == num_tickets - 1:
            # Last ticket - should be done
            assert done is True, (
                f"Episode should be done after processing last ticket (ticket {ticket_idx + 1}/{num_tickets})"
            )
        else:
            # Not last ticket - should not be done
            assert done is False, (
                f"Episode should not be done before processing all tickets "
                f"(ticket {ticket_idx + 1}/{num_tickets})"
            )
        
        # If done, stop the loop (can't step further)
        if done:
            break


# ---------------------------------------------------------------------------
# Property 11: Episode Termination
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 11: Episode Termination
@given(task=task_strategy(), action=action_strategy())
@settings(max_examples=100)
def test_episode_termination(task, action):
    """
    Property 11: Episode Termination
    
    For any episode, when _current_ticket_idx reaches the length of the 
    ticket list after a step, the done flag should be True and the 
    observation should have empty ticket_id, subject, and body fields.
    
    **Validates: Requirements 9.12, 10.1**
    """
    # Create environment and reset
    env = SupportEnvironment(task=task)
    env.reset()
    
    # Get the number of tickets for this task
    num_tickets = len(TASK_TICKETS[task])
    
    # Process all tickets
    for ticket_idx in range(num_tickets):
        # Perform step
        obs, reward, done, info = env.step(action)
        
        # Check if this is the last ticket
        if ticket_idx == num_tickets - 1:
            # Last ticket - episode should be done
            assert done is True, (
                f"Episode should be done after processing last ticket "
                f"(ticket {ticket_idx + 1}/{num_tickets})"
            )
            
            # Verify _current_ticket_idx has reached the length of ticket list
            assert env._current_ticket_idx == num_tickets, (
                f"_current_ticket_idx should equal ticket list length when done. "
                f"Expected: {num_tickets}, Got: {env._current_ticket_idx}"
            )
            
            # Verify observation has empty ticket fields
            assert obs.ticket_id == "", (
                f"Terminal observation should have empty ticket_id. "
                f"Got: {obs.ticket_id!r}"
            )
            
            assert obs.subject == "", (
                f"Terminal observation should have empty subject. "
                f"Got: {obs.subject!r}"
            )
            
            assert obs.body == "", (
                f"Terminal observation should have empty body. "
                f"Got: {obs.body!r}"
            )
            
            # Verify done flag is set in observation
            assert obs.done is True, (
                f"Terminal observation should have done=True. "
                f"Got: {obs.done}"
            )
        else:
            # Not last ticket - should not be done
            assert done is False, (
                f"Episode should not be done before processing all tickets "
                f"(ticket {ticket_idx + 1}/{num_tickets})"
            )
            
            # Verify observation has non-empty ticket fields
            assert obs.ticket_id != "", (
                f"Non-terminal observation should have non-empty ticket_id "
                f"(ticket {ticket_idx + 1}/{num_tickets})"
            )
            
            assert obs.done is False, (
                f"Non-terminal observation should have done=False "
                f"(ticket {ticket_idx + 1}/{num_tickets})"
            )


# ---------------------------------------------------------------------------
# Property 12: Reset Initialization
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 12: Reset Initialization
@given(task=task_strategy())
@settings(max_examples=100)
def test_reset_initialization(task):
    """
    Property 12: Reset Initialization
    
    For any task difficulty, calling reset() should set _current_ticket_idx 
    to 0, _episode_started to True, create a fresh episode_id, set all state 
    counters to 0, and compute escalations_needed as the count of tickets 
    with gt_escalate=True.
    
    **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9**
    """
    # Create environment
    env = SupportEnvironment(task=task)
    
    # Call reset
    obs = env.reset()
    
    # Verify _current_ticket_idx is 0
    assert env._current_ticket_idx == 0, (
        f"_current_ticket_idx should be 0 after reset. "
        f"Got: {env._current_ticket_idx}"
    )
    
    # Verify _episode_started is True
    assert env._episode_started is True, (
        f"_episode_started should be True after reset. "
        f"Got: {env._episode_started}"
    )
    
    # Verify episode_id is a fresh UUID4 (non-empty string)
    assert env.state.episode_id != "", (
        f"episode_id should be a non-empty string after reset. "
        f"Got: {env.state.episode_id!r}"
    )
    
    # Verify episode_id is a valid UUID format (36 characters with hyphens)
    assert len(env.state.episode_id) == 36, (
        f"episode_id should be 36 characters (UUID4 format). "
        f"Got length: {len(env.state.episode_id)}, value: {env.state.episode_id!r}"
    )
    
    # Verify all state counters are 0
    assert env.state.step_count == 0, (
        f"step_count should be 0 after reset. "
        f"Got: {env.state.step_count}"
    )
    
    assert env.state.total_reward == 0.0, (
        f"total_reward should be 0.0 after reset. "
        f"Got: {env.state.total_reward}"
    )
    
    assert env.state.tickets_processed == 0, (
        f"tickets_processed should be 0 after reset. "
        f"Got: {env.state.tickets_processed}"
    )
    
    assert env.state.correct_categories == 0, (
        f"correct_categories should be 0 after reset. "
        f"Got: {env.state.correct_categories}"
    )
    
    assert env.state.correct_priorities == 0, (
        f"correct_priorities should be 0 after reset. "
        f"Got: {env.state.correct_priorities}"
    )
    
    assert env.state.escalations_made == 0, (
        f"escalations_made should be 0 after reset. "
        f"Got: {env.state.escalations_made}"
    )
    
    # Verify escalations_needed is computed correctly
    # Count tickets with gt_escalate=True
    expected_escalations_needed = sum(1 for t in TASK_TICKETS[task] if t["gt_escalate"])
    
    assert env.state.escalations_needed == expected_escalations_needed, (
        f"escalations_needed should equal count of tickets with gt_escalate=True. "
        f"Expected: {expected_escalations_needed}, Got: {env.state.escalations_needed}"
    )
    
    # Verify task_name is set correctly
    assert env.state.task_name == task, (
        f"task_name should match the task parameter. "
        f"Expected: {task!r}, Got: {env.state.task_name!r}"
    )
    
    # Verify observation is returned with first ticket
    assert obs.ticket_id != "", (
        f"First observation should have non-empty ticket_id. "
        f"Got: {obs.ticket_id!r}"
    )
    
    assert obs.done is False, (
        f"First observation should have done=False. "
        f"Got: {obs.done}"
    )


# ---------------------------------------------------------------------------
# Property 13: Correct Category Tracking
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 13: Correct Category Tracking
@given(task=task_strategy(), action=action_strategy())
@settings(max_examples=100)
def test_correct_category_tracking(task, action):
    """
    Property 13: Correct Category Tracking
    
    For any step where the category component score equals 1.0, the state's 
    correct_categories counter should be incremented by 1.
    
    **Validates: Requirements 9.7**
    """
    from envs.support_env.grader import grade
    
    # Create environment and reset
    env = SupportEnvironment(task=task)
    env.reset()
    
    # Get the number of tickets for this task
    num_tickets = len(TASK_TICKETS[task])
    
    # Track expected correct_categories count
    expected_correct_categories = 0
    
    # Process all tickets
    for ticket_idx in range(num_tickets):
        # Get current ticket
        current_ticket = TASK_TICKETS[task][ticket_idx]
        
        # Capture correct_categories before step
        correct_categories_before = env.state.correct_categories
        
        # Perform step
        obs, reward, done, info = env.step(action)
        
        # Get the breakdown from info
        breakdown = info.get("breakdown", {})
        
        # Check if category score is 1.0
        category_score = breakdown.get("category", 0.0)
        
        if category_score == 1.0:
            # Category was correct - counter should increment
            expected_correct_categories += 1
            
            assert env.state.correct_categories == correct_categories_before + 1, (
                f"correct_categories should increment by 1 when category score is 1.0. "
                f"Before: {correct_categories_before}, After: {env.state.correct_categories}, "
                f"Category score: {category_score:.2f}, "
                f"Action category: {action.category!r}, GT category: {current_ticket['gt_category']!r}"
            )
        else:
            # Category was incorrect - counter should not change
            assert env.state.correct_categories == correct_categories_before, (
                f"correct_categories should not change when category score is not 1.0. "
                f"Before: {correct_categories_before}, After: {env.state.correct_categories}, "
                f"Category score: {category_score:.2f}, "
                f"Action category: {action.category!r}, GT category: {current_ticket['gt_category']!r}"
            )
        
        # Verify the running total matches our expectation
        assert env.state.correct_categories == expected_correct_categories, (
            f"correct_categories total mismatch after ticket {ticket_idx + 1}. "
            f"Expected: {expected_correct_categories}, Actual: {env.state.correct_categories}"
        )
        
        # If done, stop the loop
        if done:
            break


# ---------------------------------------------------------------------------
# Property 14: Correct Priority Tracking
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 14: Correct Priority Tracking
@given(task=task_strategy(), action=action_strategy())
@settings(max_examples=100)
def test_correct_priority_tracking(task, action):
    """
    Property 14: Correct Priority Tracking
    
    For any step where the priority component score equals 1.0, the state's 
    correct_priorities counter should be incremented by 1.
    
    **Validates: Requirements 9.8**
    """
    from envs.support_env.grader import grade
    
    # Create environment and reset
    env = SupportEnvironment(task=task)
    env.reset()
    
    # Get the number of tickets for this task
    num_tickets = len(TASK_TICKETS[task])
    
    # Track expected correct_priorities count
    expected_correct_priorities = 0
    
    # Process all tickets
    for ticket_idx in range(num_tickets):
        # Get current ticket
        current_ticket = TASK_TICKETS[task][ticket_idx]
        
        # Capture correct_priorities before step
        correct_priorities_before = env.state.correct_priorities
        
        # Perform step
        obs, reward, done, info = env.step(action)
        
        # Get the breakdown from info
        breakdown = info.get("breakdown", {})
        
        # Check if priority score is 1.0
        priority_score = breakdown.get("priority", 0.0)
        
        if priority_score == 1.0:
            # Priority was correct - counter should increment
            expected_correct_priorities += 1
            
            assert env.state.correct_priorities == correct_priorities_before + 1, (
                f"correct_priorities should increment by 1 when priority score is 1.0. "
                f"Before: {correct_priorities_before}, After: {env.state.correct_priorities}, "
                f"Priority score: {priority_score:.2f}, "
                f"Action priority: {action.priority!r}, GT priority: {current_ticket['gt_priority']!r}"
            )
        else:
            # Priority was incorrect - counter should not change
            assert env.state.correct_priorities == correct_priorities_before, (
                f"correct_priorities should not change when priority score is not 1.0. "
                f"Before: {correct_priorities_before}, After: {env.state.correct_priorities}, "
                f"Priority score: {priority_score:.2f}, "
                f"Action priority: {action.priority!r}, GT priority: {current_ticket['gt_priority']!r}"
            )
        
        # Verify the running total matches our expectation
        assert env.state.correct_priorities == expected_correct_priorities, (
            f"correct_priorities total mismatch after ticket {ticket_idx + 1}. "
            f"Expected: {expected_correct_priorities}, Actual: {env.state.correct_priorities}"
        )
        
        # If done, stop the loop
        if done:
            break



# ---------------------------------------------------------------------------
# Property 15: Escalation Action Tracking
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 15: Escalation Action Tracking
@given(task=task_strategy(), action=action_strategy())
@settings(max_examples=100)
def test_escalation_action_tracking(task, action):
    """
    Property 15: Escalation Action Tracking
    
    For any step where the action's escalate field is True, the state's 
    escalations_made counter should be incremented by 1.
    
    **Validates: Requirements 9.9**
    """
    # Create environment and reset
    env = SupportEnvironment(task=task)
    env.reset()
    
    # Get the number of tickets for this task
    num_tickets = len(TASK_TICKETS[task])
    
    # Track expected escalations_made count
    expected_escalations_made = 0
    
    # Process all tickets
    for ticket_idx in range(num_tickets):
        # Capture escalations_made before step
        escalations_made_before = env.state.escalations_made
        
        # Perform step
        obs, reward, done, info = env.step(action)
        
        # Check if action.escalate is True
        if action.escalate is True:
            # Escalation was made - counter should increment
            expected_escalations_made += 1
            
            assert env.state.escalations_made == escalations_made_before + 1, (
                f"escalations_made should increment by 1 when action.escalate is True. "
                f"Before: {escalations_made_before}, After: {env.state.escalations_made}, "
                f"Action escalate: {action.escalate}"
            )
        else:
            # No escalation - counter should not change
            assert env.state.escalations_made == escalations_made_before, (
                f"escalations_made should not change when action.escalate is False. "
                f"Before: {escalations_made_before}, After: {env.state.escalations_made}, "
                f"Action escalate: {action.escalate}"
            )
        
        # Verify the running total matches our expectation
        assert env.state.escalations_made == expected_escalations_made, (
            f"escalations_made total mismatch after ticket {ticket_idx + 1}. "
            f"Expected: {expected_escalations_made}, Actual: {env.state.escalations_made}"
        )
        
        # If done, stop the loop
        if done:
            break
