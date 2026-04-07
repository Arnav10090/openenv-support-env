"""Unit tests for grading system."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
from envs.support_env.grader import (
    _category_score, _priority_score, _escalation_score,
    _response_score, _tag_score, _penalty,
    grade_easy, grade_medium, grade_hard, grade
)
from envs.support_env.models import SupportAction


class TestComponentScorers:
    def test_category_score_correct(self):
        action = SupportAction(category="billing")
        ticket = {"gt_category": "billing"}
        assert _category_score(action, ticket) == 1.0

    def test_category_score_incorrect(self):
        action = SupportAction(category="technical")
        ticket = {"gt_category": "billing"}
        assert _category_score(action, ticket) == 0.0

    def test_priority_score_exact(self):
        action = SupportAction(priority="high")
        ticket = {"gt_priority": "high"}
        assert _priority_score(action, ticket) == 1.0

    def test_priority_score_distance_1(self):
        action = SupportAction(priority="medium")
        ticket = {"gt_priority": "high"}
        assert _priority_score(action, ticket) == 0.5

    def test_escalation_score_correct(self):
        action = SupportAction(escalate=True)
        ticket = {"gt_escalate": True}
        assert _escalation_score(action, ticket) == 1.0

    def test_response_score_full(self):
        action = SupportAction(response="We will refund the duplicate charge immediately. " * 2)
        ticket = {"gt_response_keywords": ["refund", "charge", "duplicate"]}
        assert _response_score(action, ticket) == 1.0

    def test_tag_score_all_matched(self):
        action = SupportAction(tags=["billing", "refund"])
        ticket = {"gt_tags": ["billing", "refund"]}
        assert _tag_score(action, ticket) == 1.0

    def test_penalty_empty_response(self):
        action = SupportAction(response="")
        assert _penalty(action) == 0.3


class TestDifficultyGraders:
    def test_grade_easy_perfect(self):
        action = SupportAction(
            category="billing", priority="high",
            response="We will refund the duplicate charge immediately. " * 2,
            tags=["billing", "refund"]
        )
        ticket = {
            "gt_category": "billing", "gt_priority": "high",
            "gt_escalate": False, "gt_tags": ["billing", "refund"],
            "gt_response_keywords": ["refund", "charge", "duplicate"]
        }
        score, breakdown = grade_easy(action, ticket)
        assert score == 1.0

    def test_grade_medium_includes_escalation(self):
        action = SupportAction()
        ticket = {
            "gt_category": "billing", "gt_priority": "high",
            "gt_escalate": False, "gt_tags": [],
            "gt_response_keywords": []
        }
        score, breakdown = grade_medium(action, ticket)
        assert "escalation" in breakdown

    def test_grade_dispatcher(self):
        action = SupportAction(category="billing")
        ticket = {
            "gt_category": "billing", "gt_priority": "high",
            "gt_escalate": False, "gt_tags": [],
            "gt_response_keywords": []
        }
        score, breakdown = grade("easy", action, ticket)
        assert "escalation" not in breakdown
