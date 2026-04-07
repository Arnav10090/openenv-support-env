"""
Unit tests for the ticket dataset.

Verifies:
- Exactly 9 tickets exist
- 3 tickets per difficulty level
- All required fields present in each ticket
- Ticket IDs follow naming convention
- TICKET_BY_ID and TASK_TICKETS derived correctly
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest
from envs.support_env.tickets import TICKETS, TICKET_BY_ID, TASK_TICKETS


class TestTicketDataset:
    """Test suite for ticket dataset structure and content."""

    def test_exactly_9_tickets(self):
        """Verify exactly 9 tickets exist in the dataset."""
        assert len(TICKETS) == 9, f"Expected 9 tickets, found {len(TICKETS)}"

    def test_3_tickets_per_difficulty(self):
        """Verify 3 tickets per difficulty level (easy, medium, hard)."""
        easy_tickets = [t for t in TICKETS if t["task"] == "easy"]
        medium_tickets = [t for t in TICKETS if t["task"] == "medium"]
        hard_tickets = [t for t in TICKETS if t["task"] == "hard"]

        assert len(easy_tickets) == 3, f"Expected 3 easy tickets, found {len(easy_tickets)}"
        assert len(medium_tickets) == 3, f"Expected 3 medium tickets, found {len(medium_tickets)}"
        assert len(hard_tickets) == 3, f"Expected 3 hard tickets, found {len(hard_tickets)}"

    def test_all_required_fields_present(self):
        """Verify all required fields are present in each ticket."""
        required_fields = [
            "ticket_id",
            "task",
            "subject",
            "body",
            "customer_history",
            "gt_category",
            "gt_priority",
            "gt_escalate",
            "gt_tags",
            "gt_response_keywords",
            "difficulty",
        ]

        for ticket in TICKETS:
            for field in required_fields:
                assert field in ticket, (
                    f"Ticket {ticket.get('ticket_id', 'UNKNOWN')} missing field '{field}'"
                )

    def test_ticket_id_naming_convention(self):
        """Verify ticket IDs follow naming convention (E001-E003, M001-M003, H001-H003)."""
        expected_ids = {
            "E001", "E002", "E003",  # Easy
            "M001", "M002", "M003",  # Medium
            "H001", "H002", "H003",  # Hard
        }

        actual_ids = {t["ticket_id"] for t in TICKETS}

        assert actual_ids == expected_ids, (
            f"Ticket IDs don't match expected convention.\n"
            f"Expected: {sorted(expected_ids)}\n"
            f"Actual: {sorted(actual_ids)}"
        )

    def test_ticket_by_id_derived_correctly(self):
        """Verify TICKET_BY_ID dict is derived correctly from TICKETS."""
        # Check all tickets are in TICKET_BY_ID
        assert len(TICKET_BY_ID) == len(TICKETS), (
            f"TICKET_BY_ID has {len(TICKET_BY_ID)} entries, expected {len(TICKETS)}"
        )

        # Check each ticket is accessible by its ID
        for ticket in TICKETS:
            ticket_id = ticket["ticket_id"]
            assert ticket_id in TICKET_BY_ID, f"Ticket {ticket_id} not in TICKET_BY_ID"
            assert TICKET_BY_ID[ticket_id] is ticket, (
                f"TICKET_BY_ID[{ticket_id}] is not the same object as the ticket in TICKETS"
            )

    def test_task_tickets_derived_correctly(self):
        """Verify TASK_TICKETS dict groups tickets by task correctly."""
        # Check all three difficulty levels exist
        assert "easy" in TASK_TICKETS, "TASK_TICKETS missing 'easy' key"
        assert "medium" in TASK_TICKETS, "TASK_TICKETS missing 'medium' key"
        assert "hard" in TASK_TICKETS, "TASK_TICKETS missing 'hard' key"

        # Check counts
        assert len(TASK_TICKETS["easy"]) == 3, (
            f"TASK_TICKETS['easy'] has {len(TASK_TICKETS['easy'])} tickets, expected 3"
        )
        assert len(TASK_TICKETS["medium"]) == 3, (
            f"TASK_TICKETS['medium'] has {len(TASK_TICKETS['medium'])} tickets, expected 3"
        )
        assert len(TASK_TICKETS["hard"]) == 3, (
            f"TASK_TICKETS['hard'] has {len(TASK_TICKETS['hard'])} tickets, expected 3"
        )

        # Check that tickets are correctly grouped
        for difficulty in ["easy", "medium", "hard"]:
            for ticket in TASK_TICKETS[difficulty]:
                assert ticket["task"] == difficulty, (
                    f"Ticket {ticket['ticket_id']} in TASK_TICKETS['{difficulty}'] "
                    f"has task='{ticket['task']}'"
                )

    def test_easy_ticket_ids(self):
        """Verify easy tickets have IDs E001, E002, E003."""
        easy_ids = {t["ticket_id"] for t in TASK_TICKETS["easy"]}
        expected_ids = {"E001", "E002", "E003"}
        assert easy_ids == expected_ids, (
            f"Easy ticket IDs don't match.\nExpected: {expected_ids}\nActual: {easy_ids}"
        )

    def test_medium_ticket_ids(self):
        """Verify medium tickets have IDs M001, M002, M003."""
        medium_ids = {t["ticket_id"] for t in TASK_TICKETS["medium"]}
        expected_ids = {"M001", "M002", "M003"}
        assert medium_ids == expected_ids, (
            f"Medium ticket IDs don't match.\nExpected: {expected_ids}\nActual: {medium_ids}"
        )

    def test_hard_ticket_ids(self):
        """Verify hard tickets have IDs H001, H002, H003."""
        hard_ids = {t["ticket_id"] for t in TASK_TICKETS["hard"]}
        expected_ids = {"H001", "H002", "H003"}
        assert hard_ids == expected_ids, (
            f"Hard ticket IDs don't match.\nExpected: {expected_ids}\nActual: {hard_ids}"
        )

    def test_field_types(self):
        """Verify field types are correct for all tickets."""
        for ticket in TICKETS:
            ticket_id = ticket["ticket_id"]

            # String fields
            assert isinstance(ticket["ticket_id"], str), f"{ticket_id}: ticket_id not str"
            assert isinstance(ticket["task"], str), f"{ticket_id}: task not str"
            assert isinstance(ticket["subject"], str), f"{ticket_id}: subject not str"
            assert isinstance(ticket["body"], str), f"{ticket_id}: body not str"
            assert isinstance(ticket["gt_category"], str), f"{ticket_id}: gt_category not str"
            assert isinstance(ticket["gt_priority"], str), f"{ticket_id}: gt_priority not str"
            assert isinstance(ticket["difficulty"], str), f"{ticket_id}: difficulty not str"

            # Boolean field
            assert isinstance(ticket["gt_escalate"], bool), f"{ticket_id}: gt_escalate not bool"

            # List fields
            assert isinstance(ticket["customer_history"], list), (
                f"{ticket_id}: customer_history not list"
            )
            assert isinstance(ticket["gt_tags"], list), f"{ticket_id}: gt_tags not list"
            assert isinstance(ticket["gt_response_keywords"], list), (
                f"{ticket_id}: gt_response_keywords not list"
            )

    def test_non_empty_required_string_fields(self):
        """Verify required string fields are non-empty."""
        for ticket in TICKETS:
            ticket_id = ticket["ticket_id"]

            assert ticket["ticket_id"], f"{ticket_id}: ticket_id is empty"
            assert ticket["task"], f"{ticket_id}: task is empty"
            assert ticket["subject"], f"{ticket_id}: subject is empty"
            assert ticket["body"], f"{ticket_id}: body is empty"
            assert ticket["gt_category"], f"{ticket_id}: gt_category is empty"
            assert ticket["gt_priority"], f"{ticket_id}: gt_priority is empty"
            assert ticket["difficulty"], f"{ticket_id}: difficulty is empty"

    def test_task_matches_difficulty(self):
        """Verify task field matches difficulty field for all tickets."""
        for ticket in TICKETS:
            ticket_id = ticket["ticket_id"]
            assert ticket["task"] == ticket["difficulty"], (
                f"{ticket_id}: task '{ticket['task']}' != difficulty '{ticket['difficulty']}'"
            )
