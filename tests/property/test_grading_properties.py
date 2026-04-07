"""
Property-based tests for the Customer Support Triage Environment grading system.

These tests use Hypothesis to generate random inputs and verify that the grading
functions satisfy their correctness properties across a wide range of scenarios.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hypothesis import given, settings, strategies as st

from envs.support_env.grader import (
    _category_score,
    _priority_score,
    _escalation_score,
    _response_score,
    _tag_score,
    VALID_CATEGORIES,
    VALID_PRIORITIES,
    PRIORITY_DISTANCE,
)
from envs.support_env.models import SupportAction


# ---------------------------------------------------------------------------
# Hypothesis Strategies
# ---------------------------------------------------------------------------

@st.composite
def action_strategy(draw):
    """Generate random SupportAction instances."""
    return SupportAction(
        category=draw(st.text(min_size=0, max_size=20)),
        priority=draw(st.sampled_from(["low", "medium", "high", "urgent", "invalid", ""])),
        response=draw(st.text(min_size=0, max_size=200)),
        escalate=draw(st.booleans()),
        tags=draw(st.lists(st.text(min_size=1, max_size=15), max_size=5)),
        metadata=draw(st.dictionaries(st.text(min_size=1, max_size=10), st.text(max_size=20), max_size=3))
    )


@st.composite
def ticket_strategy(draw):
    """Generate random ticket dictionaries with ground-truth labels."""
    return {
        "ticket_id": draw(st.text(min_size=1, max_size=10)),
        "gt_category": draw(st.sampled_from(list(VALID_CATEGORIES))),
        "gt_priority": draw(st.sampled_from(["low", "medium", "high", "urgent"])),
        "gt_escalate": draw(st.booleans()),
        "gt_tags": draw(st.lists(st.text(min_size=1, max_size=15), max_size=5)),
        "gt_response_keywords": draw(st.lists(st.text(min_size=1, max_size=15), max_size=5)),
    }


# ---------------------------------------------------------------------------
# Property 1: Category Scoring Correctness
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 1: Category Scoring Correctness
@given(action=action_strategy(), ticket=ticket_strategy())
@settings(max_examples=100)
def test_category_score_binary_match(action, ticket):
    """
    Property 1: Category Scoring Correctness
    
    For any action and ticket, when the action's category (after stripping and 
    lowercasing) matches the ticket's ground-truth category, the category 
    component score should be 1.0, otherwise 0.0.
    
    Validates: Requirements 5.1
    """
    score = _category_score(action, ticket)
    
    # Normalize the action category the same way the grader does
    normalized_category = (action.category or "").strip().lower()
    gt_category = ticket["gt_category"]
    
    # Verify binary scoring
    if normalized_category == gt_category:
        assert score == 1.0, (
            f"Expected score 1.0 for matching category. "
            f"Action category: {normalized_category!r}, GT: {gt_category!r}, Score: {score}"
        )
    else:
        assert score == 0.0, (
            f"Expected score 0.0 for non-matching category. "
            f"Action category: {normalized_category!r}, GT: {gt_category!r}, Score: {score}"
        )
    
    # Verify score is always in valid range
    assert 0.0 <= score <= 1.0, f"Score {score} is outside valid range [0.0, 1.0]"


# ---------------------------------------------------------------------------
# Property 2: Priority Scoring by Ordinal Distance
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 2: Priority Scoring by Ordinal Distance
@given(action=action_strategy(), ticket=ticket_strategy())
@settings(max_examples=100)
def test_priority_score_ordinal_distance(action, ticket):
    """
    Property 2: Priority Scoring by Ordinal Distance
    
    For any action and ticket, the priority component score should be determined 
    by the ordinal distance between predicted and ground-truth priorities: 
    distance 0 → 1.0, distance 1 → 0.5, distance 2 → 0.25, distance ≥3 → 0.0.
    
    **Validates: Requirements 5.2**
    """
    score = _priority_score(action, ticket)
    
    # Normalize the action priority the same way the grader does
    pred_priority = (action.priority or "medium").strip().lower()
    gt_priority = ticket["gt_priority"]
    
    # Calculate ordinal distance using PRIORITY_DISTANCE mapping
    # Invalid priorities default to index 1 (medium)
    pred_idx = PRIORITY_DISTANCE.get(pred_priority, 1)
    gt_idx = PRIORITY_DISTANCE.get(gt_priority, 1)
    distance = abs(pred_idx - gt_idx)
    
    # Verify score matches the distance formula
    if distance == 0:
        expected_score = 1.0
    elif distance == 1:
        expected_score = 0.5
    elif distance == 2:
        expected_score = 0.25
    else:  # distance >= 3
        expected_score = 0.0
    
    assert score == expected_score, (
        f"Expected score {expected_score} for distance {distance}. "
        f"Predicted priority: {pred_priority!r} (idx={pred_idx}), "
        f"GT priority: {gt_priority!r} (idx={gt_idx}), "
        f"Actual score: {score}"
    )
    
    # Verify score is always in valid range
    assert 0.0 <= score <= 1.0, f"Score {score} is outside valid range [0.0, 1.0]"


# ---------------------------------------------------------------------------
# Property 3: Escalation Binary Match
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 3: Escalation Binary Match
@given(action=action_strategy(), ticket=ticket_strategy())
@settings(max_examples=100)
def test_escalation_score_binary_match(action, ticket):
    """
    Property 3: Escalation Binary Match
    
    For any action and ticket, the escalation component score should be 1.0 
    if the action's escalate boolean matches the ticket's ground-truth escalate 
    value, otherwise 0.0.
    
    **Validates: Requirements 5.5**
    """
    score = _escalation_score(action, ticket)
    
    # Get the escalation values
    action_escalate = action.escalate
    gt_escalate = ticket["gt_escalate"]
    
    # Verify binary scoring
    if action_escalate == gt_escalate:
        assert score == 1.0, (
            f"Expected score 1.0 for matching escalation decision. "
            f"Action escalate: {action_escalate}, GT escalate: {gt_escalate}, Score: {score}"
        )
    else:
        assert score == 0.0, (
            f"Expected score 0.0 for non-matching escalation decision. "
            f"Action escalate: {action_escalate}, GT escalate: {gt_escalate}, Score: {score}"
        )
    
    # Verify score is always in valid range
    assert 0.0 <= score <= 1.0, f"Score {score} is outside valid range [0.0, 1.0]"


# ---------------------------------------------------------------------------
# Property 4: Response Scoring Formula
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 4: Response Scoring Formula
@given(action=action_strategy(), ticket=ticket_strategy())
@settings(max_examples=100)
def test_response_score_formula(action, ticket):
    """
    Property 4: Response Scoring Formula
    
    For any action and ticket, the response component score should equal 
    0.6 × keyword_coverage + 0.4 × length_adequacy, where keyword_coverage 
    is the fraction of ground-truth keywords found as case-insensitive 
    substrings in the response (0.5 if no keywords), and length_adequacy 
    is 1.0 if response length ≥ 80 else length/80.
    
    **Validates: Requirements 5.6, 5.7, 5.9**
    """
    score = _response_score(action, ticket)
    
    # Compute expected keyword coverage
    response = (action.response or "").lower()
    keywords = ticket.get("gt_response_keywords", [])
    
    if keywords:
        hits = sum(1 for kw in keywords if kw.lower() in response)
        keyword_coverage = hits / len(keywords)
    else:
        keyword_coverage = 0.5  # Default when no keywords specified
    
    # Compute expected length adequacy
    response_length = len(response)
    if response_length >= 80:
        length_adequacy = 1.0
    else:
        length_adequacy = response_length / 80
    
    # Compute expected score using the formula
    expected_score = 0.6 * keyword_coverage + 0.4 * length_adequacy
    
    # Verify the score matches the formula
    assert abs(score - expected_score) < 1e-9, (
        f"Response score does not match formula. "
        f"Expected: {expected_score:.6f}, Actual: {score:.6f}\n"
        f"Response length: {response_length}, Keywords: {len(keywords)}, "
        f"Keyword coverage: {keyword_coverage:.3f}, Length adequacy: {length_adequacy:.3f}"
    )
    
    # Verify score is always in valid range
    assert 0.0 <= score <= 1.0, f"Score {score} is outside valid range [0.0, 1.0]"


# ---------------------------------------------------------------------------
# Property 5: Tag Matching with Bidirectional Substring
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 5: Tag Matching with Bidirectional Substring
@given(action=action_strategy(), ticket=ticket_strategy())
@settings(max_examples=100)
def test_tag_score_bidirectional_substring(action, ticket):
    """
    Property 5: Tag Matching with Bidirectional Substring
    
    For any action and ticket, the tag component score should equal the 
    fraction of ground-truth tags that have at least one agent tag where 
    either the ground-truth tag is a substring of the agent tag OR the 
    agent tag is a substring of the ground-truth tag (both lowercased), 
    returning 1.0 if ground-truth tags are empty.
    
    **Validates: Requirements 5.10, 5.11**
    """
    score = _tag_score(action, ticket)
    
    # Get ground-truth tags (lowercased)
    gt_tags = [t.lower() for t in ticket.get("gt_tags", [])]
    
    # If no ground-truth tags, score should be 1.0
    if not gt_tags:
        assert score == 1.0, (
            f"Expected score 1.0 when no ground-truth tags specified. "
            f"Actual score: {score}"
        )
        return
    
    # Get agent tags (lowercased)
    agent_tags = [t.lower() for t in (action.tags or [])]
    
    # Count hits using bidirectional substring matching
    hits = 0
    for gt in gt_tags:
        # Check if any agent tag matches this ground-truth tag
        # Match criteria: gt in agent_tag OR agent_tag in gt
        matched = any(gt in a or a in gt for a in agent_tags)
        if matched:
            hits += 1
    
    # Compute expected score
    expected_score = hits / len(gt_tags)
    
    # Verify the score matches the expected value
    assert abs(score - expected_score) < 1e-9, (
        f"Tag score does not match expected value. "
        f"Expected: {expected_score:.6f}, Actual: {score:.6f}\n"
        f"Ground-truth tags: {gt_tags}\n"
        f"Agent tags: {agent_tags}\n"
        f"Hits: {hits}/{len(gt_tags)}"
    )
    
    # Verify score is always in valid range
    assert 0.0 <= score <= 1.0, f"Score {score} is outside valid range [0.0, 1.0]"


# ---------------------------------------------------------------------------
# Property 6: Penalty Accumulation
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 6: Penalty Accumulation
@given(action=action_strategy())
@settings(max_examples=100)
def test_penalty_accumulation(action):
    """
    Property 6: Penalty Accumulation
    
    For any action, the penalty should equal the sum of: 0.3 if response 
    is empty/whitespace-only, 0.1 if category is not in VALID_CATEGORIES, 
    0.1 if priority is not in VALID_PRIORITIES.
    
    **Validates: Requirements 5.13**
    """
    from envs.support_env.grader import _penalty
    
    penalty = _penalty(action)
    
    # Compute expected penalty
    expected_penalty = 0.0
    
    # Check empty response (0.3 penalty)
    if not (action.response or "").strip():
        expected_penalty += 0.3
    
    # Check invalid category (0.1 penalty)
    if (action.category or "").lower() not in VALID_CATEGORIES:
        expected_penalty += 0.1
    
    # Check invalid priority (0.1 penalty)
    if (action.priority or "").lower() not in VALID_PRIORITIES:
        expected_penalty += 0.1
    
    # Verify penalty matches expected value
    assert abs(penalty - expected_penalty) < 1e-9, (
        f"Penalty does not match expected value. "
        f"Expected: {expected_penalty:.2f}, Actual: {penalty:.2f}\n"
        f"Response empty: {not (action.response or '').strip()}\n"
        f"Category '{action.category}' valid: {(action.category or '').lower() in VALID_CATEGORIES}\n"
        f"Priority '{action.priority}' valid: {(action.priority or '').lower() in VALID_PRIORITIES}"
    )
    
    # Verify penalty is always non-negative
    assert penalty >= 0.0, f"Penalty {penalty} should be non-negative"
    
    # Verify penalty is at most 0.5 (0.3 + 0.1 + 0.1)
    assert penalty <= 0.5, f"Penalty {penalty} exceeds maximum of 0.5"


# ---------------------------------------------------------------------------
# Property 7: Weighted Scoring with Clamping
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 7: Weighted Scoring with Clamping
@given(action=action_strategy(), ticket=ticket_strategy())
@settings(max_examples=100)
def test_weighted_scoring_with_clamping(action, ticket):
    """
    Property 7: Weighted Scoring with Clamping
    
    For any task difficulty, action, and ticket, the final score should equal 
    max(0.0, min(1.0, weighted_sum - penalty)), where weighted_sum applies 
    the difficulty-specific weights to component scores:
    - easy: 40% category, 30% priority, 20% response, 10% tags
    - medium: 25% category, 25% priority, 25% escalation, 20% response, 5% tags
    - hard: 15% category, 20% priority, 30% escalation, 25% response, 10% tags
    
    **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
    """
    from envs.support_env.grader import grade, _penalty
    
    # Test all three difficulty levels
    for task in ["easy", "medium", "hard"]:
        score, breakdown = grade(task, action, ticket)
        
        # Compute component scores
        cat_score = _category_score(action, ticket)
        pri_score = _priority_score(action, ticket)
        esc_score = _escalation_score(action, ticket)
        res_score = _response_score(action, ticket)
        tag_score = _tag_score(action, ticket)
        penalty = _penalty(action)
        
        # Compute expected weighted sum based on difficulty
        if task == "easy":
            expected_weighted_sum = (
                0.40 * cat_score +
                0.30 * pri_score +
                0.20 * res_score +
                0.10 * tag_score
            )
        elif task == "medium":
            expected_weighted_sum = (
                0.25 * cat_score +
                0.25 * pri_score +
                0.25 * esc_score +
                0.20 * res_score +
                0.05 * tag_score
            )
        else:  # hard
            expected_weighted_sum = (
                0.15 * cat_score +
                0.20 * pri_score +
                0.30 * esc_score +
                0.25 * res_score +
                0.10 * tag_score
            )
        
        # Apply clamping: max(0.0, min(1.0, weighted_sum - penalty))
        expected_score = max(0.0, min(1.0, expected_weighted_sum - penalty))
        
        # Verify the score matches the expected value
        assert abs(score - expected_score) < 1e-9, (
            f"Score does not match weighted formula for task '{task}'. "
            f"Expected: {expected_score:.6f}, Actual: {score:.6f}\n"
            f"Weighted sum: {expected_weighted_sum:.6f}, Penalty: {penalty:.3f}\n"
            f"Components: cat={cat_score:.2f}, pri={pri_score:.2f}, "
            f"esc={esc_score:.2f}, res={res_score:.2f}, tag={tag_score:.2f}"
        )
        
        # Verify score is always in valid range [0.0, 1.0]
        assert 0.0 <= score <= 1.0, (
            f"Score {score} for task '{task}' is outside valid range [0.0, 1.0]"
        )
        
        # Verify breakdown contains correct keys
        if task == "easy":
            expected_keys = {"category", "priority", "response", "tags", "penalty"}
        else:  # medium or hard
            expected_keys = {"category", "priority", "escalation", "response", "tags", "penalty"}
        
        assert set(breakdown.keys()) == expected_keys, (
            f"Breakdown keys for task '{task}' do not match expected. "
            f"Expected: {expected_keys}, Actual: {set(breakdown.keys())}"
        )
        
        # Verify breakdown values match component scores
        assert abs(breakdown["category"] - cat_score) < 1e-9, (
            f"Breakdown category score mismatch for task '{task}'"
        )
        assert abs(breakdown["priority"] - pri_score) < 1e-9, (
            f"Breakdown priority score mismatch for task '{task}'"
        )
        assert abs(breakdown["response"] - res_score) < 1e-9, (
            f"Breakdown response score mismatch for task '{task}'"
        )
        assert abs(breakdown["tags"] - tag_score) < 1e-9, (
            f"Breakdown tags score mismatch for task '{task}'"
        )
        
        if task in ["medium", "hard"]:
            assert abs(breakdown["escalation"] - esc_score) < 1e-9, (
                f"Breakdown escalation score mismatch for task '{task}'"
            )
        
        # Verify penalty is stored as negative value in breakdown
        assert abs(breakdown["penalty"] - (-penalty)) < 1e-9, (
            f"Breakdown penalty should be negative. "
            f"Expected: {-penalty:.3f}, Actual: {breakdown['penalty']:.3f}"
        )


# ---------------------------------------------------------------------------
# Property 8: Deterministic Scoring
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 8: Deterministic Scoring
@given(action=action_strategy(), ticket=ticket_strategy())
@settings(max_examples=100)
def test_deterministic_scoring(action, ticket):
    """
    Property 8: Deterministic Scoring
    
    For any action and ticket pair, calling the grade function multiple times 
    with the same inputs should produce identical (score, breakdown) tuples 
    across all runs, demonstrating complete determinism with no random sampling.
    
    **Validates: Requirements 25.1, 25.2**
    """
    from envs.support_env.grader import grade
    
    # Test all three difficulty levels
    for task in ["easy", "medium", "hard"]:
        # Call grade() multiple times with the same inputs
        results = []
        for _ in range(5):  # Call 5 times to verify consistency
            score, breakdown = grade(task, action, ticket)
            results.append((score, breakdown))
        
        # Verify all results are identical
        first_score, first_breakdown = results[0]
        
        for i, (score, breakdown) in enumerate(results[1:], start=1):
            # Verify scores are identical
            assert score == first_score, (
                f"Score mismatch on call {i+1} for task '{task}'. "
                f"First call: {first_score:.6f}, Call {i+1}: {score:.6f}"
            )
            
            # Verify breakdown keys are identical
            assert set(breakdown.keys()) == set(first_breakdown.keys()), (
                f"Breakdown keys mismatch on call {i+1} for task '{task}'. "
                f"First call keys: {set(first_breakdown.keys())}, "
                f"Call {i+1} keys: {set(breakdown.keys())}"
            )
            
            # Verify all breakdown values are identical
            for key in first_breakdown.keys():
                assert breakdown[key] == first_breakdown[key], (
                    f"Breakdown['{key}'] mismatch on call {i+1} for task '{task}'. "
                    f"First call: {first_breakdown[key]:.6f}, "
                    f"Call {i+1}: {breakdown[key]:.6f}"
                )


# ---------------------------------------------------------------------------
# Property 9: Score Range Bounds
# ---------------------------------------------------------------------------

# Feature: customer-support-triage-openenv, Property 9: Score Range Bounds
@given(action=action_strategy(), ticket=ticket_strategy())
@settings(max_examples=100)
def test_score_range_bounds(action, ticket):
    """
    Property 9: Score Range Bounds
    
    For any action and ticket, the final score returned by any grading function 
    should always be in the range [0.0, 1.0] inclusive, regardless of component 
    scores or penalties.
    
    **Validates: Requirements 6.5, 6.12**
    """
    from envs.support_env.grader import grade, grade_easy, grade_medium, grade_hard
    
    # Test all three difficulty levels using the main grade() function
    for task in ["easy", "medium", "hard"]:
        score, breakdown = grade(task, action, ticket)
        
        # Verify score is in valid range [0.0, 1.0]
        assert 0.0 <= score <= 1.0, (
            f"Score {score:.6f} for task '{task}' is outside valid range [0.0, 1.0]. "
            f"Action: category={action.category!r}, priority={action.priority!r}, "
            f"escalate={action.escalate}, response_len={len(action.response or '')}, "
            f"tags={action.tags}\n"
            f"Breakdown: {breakdown}"
        )
        
        # Verify score is exactly 0.0 or 1.0 or in between (no values outside)
        assert score >= 0.0, f"Score {score:.6f} is below 0.0 for task '{task}'"
        assert score <= 1.0, f"Score {score:.6f} is above 1.0 for task '{task}'"
    
    # Also test individual grading functions directly
    for grader_name, grader_func in [("easy", grade_easy), ("medium", grade_medium), ("hard", grade_hard)]:
        score, breakdown = grader_func(action, ticket)
        
        # Verify score is in valid range [0.0, 1.0]
        assert 0.0 <= score <= 1.0, (
            f"Score {score:.6f} from {grader_name} grader is outside valid range [0.0, 1.0]. "
            f"Action: category={action.category!r}, priority={action.priority!r}, "
            f"escalate={action.escalate}, response_len={len(action.response or '')}, "
            f"tags={action.tags}\n"
            f"Breakdown: {breakdown}"
        )
        
        # Verify the clamping is working correctly
        # Even if component scores and penalties would push outside [0.0, 1.0],
        # the final score should be clamped
        assert score >= 0.0, f"Score {score:.6f} is below 0.0 from {grader_name} grader"
        assert score <= 1.0, f"Score {score:.6f} is above 1.0 from {grader_name} grader"
