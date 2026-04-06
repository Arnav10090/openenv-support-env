"""
Ticket dataset for the Customer Support Triage Environment.

Each ticket has:
  - content fields (subject, body, customer_history)
  - ground truth labels (category, priority, should_escalate, key_tags)
  - task assignment (easy / medium / hard)

Grading is deterministic and reproducible.
"""

from __future__ import annotations
from typing import Any, Dict, List

TICKETS: List[Dict[str, Any]] = [

    # -----------------------------------------------------------------------
    # EASY TASK  (clear signals, unambiguous category + priority)
    # -----------------------------------------------------------------------
    {
        "ticket_id": "E001",
        "task": "easy",
        "subject": "I was charged twice for my subscription",
        "body": (
            "Hello, I just noticed that my credit card was charged twice this month "
            "for my Premium subscription — once on April 1st and again on April 3rd. "
            "Both charges are for $29.99. Please refund the duplicate charge as soon as possible."
        ),
        "customer_history": [],
        "gt_category": "billing",
        "gt_priority": "high",
        "gt_escalate": False,
        "gt_tags": ["duplicate-charge", "refund"],
        "gt_response_keywords": ["refund", "apologize", "billing"],
        "difficulty": "easy",
    },
    {
        "ticket_id": "E002",
        "task": "easy",
        "subject": "Cannot log into my account",
        "body": (
            "Hi, I've been trying to log in to my account for the past hour but keep "
            "getting the error 'Invalid credentials'. I am sure my password is correct "
            "because I just reset it 10 minutes ago. Please help!"
        ),
        "customer_history": [],
        "gt_category": "account",
        "gt_priority": "high",
        "gt_escalate": False,
        "gt_tags": ["login", "credentials"],
        "gt_response_keywords": ["login", "password", "account"],
        "difficulty": "easy",
    },
    {
        "ticket_id": "E003",
        "task": "easy",
        "subject": "Where is my order? #ORD-4821",
        "body": (
            "I placed an order 7 days ago (order #ORD-4821) and it still hasn't arrived. "
            "The tracking page says 'In transit' but hasn't updated in 5 days. "
            "Can you tell me what is happening?"
        ),
        "customer_history": [],
        "gt_category": "shipping",
        "gt_priority": "medium",
        "gt_escalate": False,
        "gt_tags": ["tracking", "delayed-shipment"],
        "gt_response_keywords": ["order", "tracking", "shipping"],
        "difficulty": "easy",
    },

    # -----------------------------------------------------------------------
    # MEDIUM TASK  (moderate ambiguity, escalation judgment needed)
    # -----------------------------------------------------------------------
    {
        "ticket_id": "M001",
        "task": "medium",
        "subject": "App crashes every time I try to export data",
        "body": (
            "Your export feature has been broken for me for two weeks now. "
            "Every time I click 'Export to CSV', the app just freezes and then crashes. "
            "I've tried reinstalling — same issue. I run a business on this tool and "
            "I have a client presentation tomorrow. This is completely unacceptable."
        ),
        "customer_history": [
            {
                "ticket_id": "M001-prev",
                "subject": "Export feature slow",
                "resolution": "asked to reinstall",
                "date": "2026-03-22",
            }
        ],
        "gt_category": "technical",
        "gt_priority": "urgent",
        "gt_escalate": True,
        "gt_tags": ["crash", "export", "repeat-contact", "business-impact"],
        "gt_response_keywords": ["escalate", "urgent", "engineer", "sorry"],
        "difficulty": "medium",
    },
    {
        "ticket_id": "M002",
        "task": "medium",
        "subject": "I want to cancel my subscription",
        "body": (
            "I'd like to cancel my annual subscription immediately and get a full refund. "
            "I signed up 3 days ago and realized this product doesn't meet my needs. "
            "I saw on your website that there's a 7-day money-back guarantee."
        ),
        "customer_history": [],
        "gt_category": "billing",
        "gt_priority": "high",
        "gt_escalate": False,
        "gt_tags": ["cancellation", "refund", "money-back-guarantee"],
        "gt_response_keywords": ["refund", "cancel", "guarantee", "process"],
        "difficulty": "medium",
    },
    {
        "ticket_id": "M003",
        "task": "medium",
        "subject": "My data was deleted without warning",
        "body": (
            "All the projects I had in my workspace are GONE. I logged in this morning "
            "and everything is missing. I did NOT delete anything. I have 3 years of "
            "work in there. Please tell me what happened and restore my data immediately."
        ),
        "customer_history": [],
        "gt_category": "technical",
        "gt_priority": "urgent",
        "gt_escalate": True,
        "gt_tags": ["data-loss", "urgent", "restore"],
        "gt_response_keywords": ["restore", "escalate", "data", "sorry", "urgent"],
        "difficulty": "medium",
    },

    # -----------------------------------------------------------------------
    # HARD TASK  (complex, multi-issue, nuanced escalation + tone)
    # -----------------------------------------------------------------------
    {
        "ticket_id": "H001",
        "task": "hard",
        "subject": "Legal threat – GDPR data breach",
        "body": (
            "I am writing to formally notify you that I believe your company has "
            "suffered a data breach affecting my personal information. I received "
            "phishing emails using my account email which I use exclusively for "
            "your service. I am a EU citizen and intend to file a complaint with "
            "my national DPA unless you provide a full incident report within 72 hours. "
            "My solicitor will be in contact."
        ),
        "customer_history": [
            {
                "ticket_id": "H001-prev1",
                "subject": "Data export request",
                "resolution": "export sent",
                "date": "2026-01-10",
            }
        ],
        "gt_category": "account",
        "gt_priority": "urgent",
        "gt_escalate": True,
        "gt_tags": ["gdpr", "data-breach", "legal", "compliance", "sensitive"],
        "gt_response_keywords": ["escalate", "legal", "data", "privacy", "sorry", "urgent"],
        "difficulty": "hard",
    },
    {
        "ticket_id": "H002",
        "task": "hard",
        "subject": "Fraudulent charges + account hacked + need refund NOW",
        "body": (
            "Someone hacked my account. They changed my email and phone number and "
            "made 4 purchases totaling $487. I've already contacted my bank. "
            "I need: 1) my account secured immediately, 2) all fraudulent charges "
            "reversed, 3) explanation of how this happened. "
            "I am furious and will post about this publicly if not resolved today."
        ),
        "customer_history": [
            {
                "ticket_id": "H002-prev1",
                "subject": "Can't receive 2FA codes",
                "resolution": "advised to check phone settings",
                "date": "2026-03-30",
            }
        ],
        "gt_category": "account",
        "gt_priority": "urgent",
        "gt_escalate": True,
        "gt_tags": ["fraud", "hacked", "refund", "security", "multi-issue", "angry-customer"],
        "gt_response_keywords": ["secure", "fraud", "refund", "escalate", "apologize", "investigate"],
        "difficulty": "hard",
    },
    {
        "ticket_id": "H003",
        "task": "hard",
        "subject": "API rate limits destroying our production system",
        "body": (
            "We are an enterprise customer (contract #ENT-2291) paying $2,400/month. "
            "Since your API update on April 4th, our rate limits have been cut in half "
            "with zero notice. This has taken down our production service affecting "
            "50,000 end users. We need: immediate rate limit restoration, "
            "SLA breach acknowledgment, and a compensation credit. "
            "Our CTO is already on the phone with your sales team."
        ),
        "customer_history": [
            {
                "ticket_id": "H003-prev1",
                "subject": "Enterprise onboarding",
                "resolution": "onboarded successfully",
                "date": "2025-09-01",
            },
            {
                "ticket_id": "H003-prev2",
                "subject": "API limits question",
                "resolution": "confirmed limits",
                "date": "2026-02-15",
            },
        ],
        "gt_category": "technical",
        "gt_priority": "urgent",
        "gt_escalate": True,
        "gt_tags": ["enterprise", "api", "sla", "production-down", "rate-limit", "compensation"],
        "gt_response_keywords": [
            "escalate", "enterprise", "sla", "restore", "apologize", "compensat", "urgent"
        ],
        "difficulty": "hard",
    },
]

# Build a lookup by ticket_id
TICKET_BY_ID: Dict[str, Dict[str, Any]] = {t["ticket_id"]: t for t in TICKETS}

# Group by task difficulty
TASK_TICKETS: Dict[str, List[Dict[str, Any]]] = {
    "easy": [t for t in TICKETS if t["task"] == "easy"],
    "medium": [t for t in TICKETS if t["task"] == "medium"],
    "hard": [t for t in TICKETS if t["task"] == "hard"],
}
