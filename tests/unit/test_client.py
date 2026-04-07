"""
Unit tests for SupportEnvClient HTTP client.

Tests verify:
- reset() returns SupportObservation
- step() returns correct dict structure
- state() returns SupportState
- health() returns correct structure
- Connection error handling
- Observation parsing with missing fields
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from urllib.error import URLError

from envs.support_env.client import SupportEnvClient
from envs.support_env.models import SupportAction, SupportObservation, SupportState


class TestSupportEnvClientInit:
    """Test SupportEnvClient initialization."""

    def test_default_base_url(self):
        """Test that default base_url is set correctly."""
        client = SupportEnvClient()
        assert client.base_url == "http://localhost:8000"

    def test_custom_base_url(self):
        """Test that custom base_url is set correctly."""
        client = SupportEnvClient(base_url="http://example.com:7860")
        assert client.base_url == "http://example.com:7860"

    def test_trailing_slash_stripped(self):
        """Test that trailing slash is stripped from base_url."""
        client = SupportEnvClient(base_url="http://example.com:7860/")
        assert client.base_url == "http://example.com:7860"

    def test_multiple_trailing_slashes_stripped(self):
        """Test that multiple trailing slashes are stripped."""
        client = SupportEnvClient(base_url="http://example.com:7860///")
        assert client.base_url == "http://example.com:7860"


class TestSupportEnvClientReset:
    """Test SupportEnvClient.reset() method."""

    @patch('envs.support_env.client.urlopen')
    def test_reset_returns_support_observation(self, mock_urlopen):
        """Test that reset() returns a SupportObservation instance."""
        # Mock response data
        response_data = {
            "ticket_id": "E001",
            "subject": "Test ticket",
            "body": "Test body",
            "customer_history": [],
            "available_categories": ["billing", "technical", "account", "shipping", "general"],
            "available_priorities": ["low", "medium", "high", "urgent"],
            "step_feedback": "Episode started. Good luck!",
            "reward": 0.0,
            "done": False,
            "metadata": {}
        }
        
        # Setup mock
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        client = SupportEnvClient()
        obs = client.reset()
        
        assert isinstance(obs, SupportObservation)
        assert obs.ticket_id == "E001"
        assert obs.subject == "Test ticket"
        assert obs.body == "Test body"
        assert obs.step_feedback == "Episode started. Good luck!"
        assert obs.reward == 0.0
        assert obs.done is False

    @patch('envs.support_env.client.urlopen')
    def test_reset_calls_post_endpoint(self, mock_urlopen):
        """Test that reset() calls POST /reset endpoint."""
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "ticket_id": "E001",
            "subject": "Test",
            "body": "Test",
            "customer_history": [],
            "available_categories": ["billing", "technical", "account", "shipping", "general"],
            "available_priorities": ["low", "medium", "high", "urgent"],
            "step_feedback": "",
            "reward": 0.0,
            "done": False,
            "metadata": {}
        }).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        client = SupportEnvClient(base_url="http://test.com")
        client.reset()
        
        # Verify urlopen was called
        assert mock_urlopen.called
        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "http://test.com/reset"
        assert request.get_method() == "POST"


class TestSupportEnvClientStep:
    """Test SupportEnvClient.step() method."""

    @patch('envs.support_env.client.urlopen')
    def test_step_returns_correct_dict_structure(self, mock_urlopen):
        """Test that step() returns dict with observation, reward, done, info keys."""
        # Mock response data
        response_data = {
            "observation": {
                "ticket_id": "E002",
                "subject": "Next ticket",
                "body": "Next body",
                "customer_history": [],
                "available_categories": ["billing", "technical", "account", "shipping", "general"],
                "available_priorities": ["low", "medium", "high", "urgent"],
                "step_feedback": "Reward: 0.85",
                "reward": 0.85,
                "done": False,
                "metadata": {}
            },
            "reward": 0.85,
            "done": False,
            "info": {"breakdown": {"category": 1.0, "priority": 1.0}, "ticket_id": "E001"}
        }
        
        # Setup mock
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        client = SupportEnvClient()
        action = SupportAction(category="billing", priority="high", response="Test response")
        result = client.step(action)
        
        # Verify structure
        assert isinstance(result, dict)
        assert "observation" in result
        assert "reward" in result
        assert "done" in result
        assert "info" in result
        
        # Verify types
        assert isinstance(result["observation"], SupportObservation)
        assert isinstance(result["reward"], float)
        assert isinstance(result["done"], bool)
        assert isinstance(result["info"], dict)
        
        # Verify values
        assert result["reward"] == 0.85
        assert result["done"] is False
        assert result["info"]["ticket_id"] == "E001"

    @patch('envs.support_env.client.urlopen')
    def test_step_sends_action_payload(self, mock_urlopen):
        """Test that step() sends correct action payload."""
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "observation": {
                "ticket_id": "",
                "subject": "",
                "body": "",
                "customer_history": [],
                "available_categories": ["billing", "technical", "account", "shipping", "general"],
                "available_priorities": ["low", "medium", "high", "urgent"],
                "step_feedback": "",
                "reward": 0.0,
                "done": False,
                "metadata": {}
            },
            "reward": 0.0,
            "done": False,
            "info": {}
        }).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        client = SupportEnvClient()
        action = SupportAction(
            category="billing",
            priority="high",
            response="We apologize for the inconvenience",
            escalate=True,
            tags=["refund", "urgent"],
            metadata={"source": "test"}
        )
        client.step(action)
        
        # Verify request
        assert mock_urlopen.called
        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "http://localhost:8000/step"
        assert request.get_method() == "POST"
        
        # Verify payload
        sent_data = json.loads(request.data.decode('utf-8'))
        assert sent_data["category"] == "billing"
        assert sent_data["priority"] == "high"
        assert sent_data["response"] == "We apologize for the inconvenience"
        assert sent_data["escalate"] is True
        assert sent_data["tags"] == ["refund", "urgent"]
        assert sent_data["metadata"] == {"source": "test"}

    @patch('envs.support_env.client.urlopen')
    def test_step_handles_missing_info_field(self, mock_urlopen):
        """Test that step() handles missing info field in response."""
        # Mock response without info field
        response_data = {
            "observation": {
                "ticket_id": "E002",
                "subject": "Test",
                "body": "Test",
                "customer_history": [],
                "available_categories": ["billing", "technical", "account", "shipping", "general"],
                "available_priorities": ["low", "medium", "high", "urgent"],
                "step_feedback": "",
                "reward": 0.5,
                "done": False,
                "metadata": {}
            },
            "reward": 0.5,
            "done": False
            # Note: no "info" field
        }
        
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        client = SupportEnvClient()
        action = SupportAction()
        result = client.step(action)
        
        # Should default to empty dict
        assert result["info"] == {}


class TestSupportEnvClientState:
    """Test SupportEnvClient.state() method."""

    @patch('envs.support_env.client.urlopen')
    def test_state_returns_support_state(self, mock_urlopen):
        """Test that state() returns a SupportState instance."""
        # Mock response data
        response_data = {
            "episode_id": "abc-123",
            "task_name": "medium",
            "step_count": 2,
            "total_reward": 1.5,
            "tickets_processed": 2,
            "correct_categories": 2,
            "correct_priorities": 1,
            "escalations_made": 1,
            "escalations_needed": 2,
            "metadata": {}
        }
        
        # Setup mock
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        client = SupportEnvClient()
        state = client.state()
        
        assert isinstance(state, SupportState)
        assert state.episode_id == "abc-123"
        assert state.task_name == "medium"
        assert state.step_count == 2
        assert state.total_reward == 1.5
        assert state.tickets_processed == 2
        assert state.correct_categories == 2
        assert state.correct_priorities == 1
        assert state.escalations_made == 1
        assert state.escalations_needed == 2

    @patch('envs.support_env.client.urlopen')
    def test_state_calls_get_endpoint(self, mock_urlopen):
        """Test that state() calls GET /state endpoint."""
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "episode_id": "test",
            "task_name": "easy",
            "step_count": 0,
            "total_reward": 0.0,
            "tickets_processed": 0,
            "correct_categories": 0,
            "correct_priorities": 0,
            "escalations_made": 0,
            "escalations_needed": 0,
            "metadata": {}
        }).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        client = SupportEnvClient(base_url="http://test.com")
        client.state()
        
        # Verify urlopen was called
        assert mock_urlopen.called
        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "http://test.com/state"
        assert request.get_method() == "GET"

    @patch('envs.support_env.client.urlopen')
    def test_state_handles_missing_fields(self, mock_urlopen):
        """Test that state() provides defaults for missing fields."""
        # Mock response with minimal data
        response_data = {
            "episode_id": "test-123"
            # All other fields missing
        }
        
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        client = SupportEnvClient()
        state = client.state()
        
        # Should use defaults
        assert state.episode_id == "test-123"
        assert state.task_name == "easy"
        assert state.step_count == 0
        assert state.total_reward == 0.0
        assert state.tickets_processed == 0
        assert state.correct_categories == 0
        assert state.correct_priorities == 0
        assert state.escalations_made == 0
        assert state.escalations_needed == 0


class TestSupportEnvClientHealth:
    """Test SupportEnvClient.health() method."""

    @patch('envs.support_env.client.urlopen')
    def test_health_returns_correct_structure(self, mock_urlopen):
        """Test that health() returns dict with status, task, version."""
        # Mock response data
        response_data = {
            "status": "healthy",
            "task": "easy",
            "version": "1.0.0"
        }
        
        # Setup mock
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        client = SupportEnvClient()
        health = client.health()
        
        assert isinstance(health, dict)
        assert health["status"] == "healthy"
        assert health["task"] == "easy"
        assert health["version"] == "1.0.0"

    @patch('envs.support_env.client.urlopen')
    def test_health_calls_get_endpoint(self, mock_urlopen):
        """Test that health() calls GET /health endpoint."""
        # Mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "healthy",
            "task": "medium",
            "version": "1.0.0"
        }).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        client = SupportEnvClient(base_url="http://test.com")
        client.health()
        
        # Verify urlopen was called
        assert mock_urlopen.called
        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "http://test.com/health"
        assert request.get_method() == "GET"


class TestSupportEnvClientConnectionErrors:
    """Test SupportEnvClient connection error handling."""

    @patch('envs.support_env.client.urlopen')
    def test_reset_raises_connection_error_on_urlerror(self, mock_urlopen):
        """Test that reset() raises ConnectionError when URLError occurs."""
        mock_urlopen.side_effect = URLError("Connection refused")
        
        client = SupportEnvClient()
        
        with pytest.raises(ConnectionError) as exc_info:
            client.reset()
        
        assert "Cannot reach environment at" in str(exc_info.value)
        assert "http://localhost:8000/reset" in str(exc_info.value)

    @patch('envs.support_env.client.urlopen')
    def test_step_raises_connection_error_on_urlerror(self, mock_urlopen):
        """Test that step() raises ConnectionError when URLError occurs."""
        mock_urlopen.side_effect = URLError("Connection refused")
        
        client = SupportEnvClient()
        action = SupportAction()
        
        with pytest.raises(ConnectionError) as exc_info:
            client.step(action)
        
        assert "Cannot reach environment at" in str(exc_info.value)
        assert "http://localhost:8000/step" in str(exc_info.value)

    @patch('envs.support_env.client.urlopen')
    def test_state_raises_connection_error_on_urlerror(self, mock_urlopen):
        """Test that state() raises ConnectionError when URLError occurs."""
        mock_urlopen.side_effect = URLError("Connection refused")
        
        client = SupportEnvClient()
        
        with pytest.raises(ConnectionError) as exc_info:
            client.state()
        
        assert "Cannot reach environment at" in str(exc_info.value)
        assert "http://localhost:8000/state" in str(exc_info.value)

    @patch('envs.support_env.client.urlopen')
    def test_health_raises_connection_error_on_urlerror(self, mock_urlopen):
        """Test that health() raises ConnectionError when URLError occurs."""
        mock_urlopen.side_effect = URLError("Connection refused")
        
        client = SupportEnvClient()
        
        with pytest.raises(ConnectionError) as exc_info:
            client.health()
        
        assert "Cannot reach environment at" in str(exc_info.value)
        assert "http://localhost:8000/health" in str(exc_info.value)


class TestSupportEnvClientObservationParsing:
    """Test SupportEnvClient._parse_observation() with missing fields."""

    def test_parse_observation_with_all_fields(self):
        """Test parsing observation with all fields present."""
        data = {
            "ticket_id": "E001",
            "subject": "Test subject",
            "body": "Test body",
            "customer_history": [{"ticket_id": "T001", "subject": "Previous"}],
            "available_categories": ["billing", "technical"],
            "available_priorities": ["low", "high"],
            "step_feedback": "Good job",
            "reward": 0.9,
            "done": True,
            "metadata": {"difficulty": "easy"}
        }
        
        obs = SupportEnvClient._parse_observation(data)
        
        assert obs.ticket_id == "E001"
        assert obs.subject == "Test subject"
        assert obs.body == "Test body"
        assert obs.customer_history == [{"ticket_id": "T001", "subject": "Previous"}]
        assert obs.available_categories == ["billing", "technical"]
        assert obs.available_priorities == ["low", "high"]
        assert obs.step_feedback == "Good job"
        assert obs.reward == 0.9
        assert obs.done is True
        assert obs.metadata == {"difficulty": "easy"}

    def test_parse_observation_with_missing_fields(self):
        """Test parsing observation with missing fields uses defaults."""
        data = {
            "ticket_id": "E001"
            # All other fields missing
        }
        
        obs = SupportEnvClient._parse_observation(data)
        
        assert obs.ticket_id == "E001"
        assert obs.subject == ""
        assert obs.body == ""
        assert obs.customer_history == []
        assert obs.available_categories == ["billing", "technical", "account", "shipping", "general"]
        assert obs.available_priorities == ["low", "medium", "high", "urgent"]
        assert obs.step_feedback == ""
        assert obs.reward == 0.0
        assert obs.done is False
        assert obs.metadata == {}

    def test_parse_observation_with_empty_dict(self):
        """Test parsing observation with empty dict uses all defaults."""
        data = {}
        
        obs = SupportEnvClient._parse_observation(data)
        
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

    def test_parse_observation_with_partial_fields(self):
        """Test parsing observation with some fields present."""
        data = {
            "ticket_id": "M002",
            "subject": "Partial ticket",
            "reward": 0.75,
            "done": False
            # Other fields missing
        }
        
        obs = SupportEnvClient._parse_observation(data)
        
        assert obs.ticket_id == "M002"
        assert obs.subject == "Partial ticket"
        assert obs.body == ""
        assert obs.customer_history == []
        assert obs.reward == 0.75
        assert obs.done is False
        assert obs.metadata == {}


class TestSupportEnvClientRequestHeaders:
    """Test that HTTP requests include correct headers."""

    @patch('envs.support_env.client.urlopen')
    def test_post_request_has_content_type_header(self, mock_urlopen):
        """Test that POST requests include Content-Type: application/json header."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "ticket_id": "",
            "subject": "",
            "body": "",
            "customer_history": [],
            "available_categories": ["billing", "technical", "account", "shipping", "general"],
            "available_priorities": ["low", "medium", "high", "urgent"],
            "step_feedback": "",
            "reward": 0.0,
            "done": False,
            "metadata": {}
        }).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        client = SupportEnvClient()
        client.reset()
        
        request = mock_urlopen.call_args[0][0]
        assert request.get_header('Content-type') == 'application/json'

    @patch('envs.support_env.client.urlopen')
    def test_request_has_timeout(self, mock_urlopen):
        """Test that requests include 30-second timeout."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "status": "healthy",
            "task": "easy",
            "version": "1.0.0"
        }).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response
        
        client = SupportEnvClient()
        client.health()
        
        # Verify timeout parameter
        assert mock_urlopen.call_args[1]['timeout'] == 30
