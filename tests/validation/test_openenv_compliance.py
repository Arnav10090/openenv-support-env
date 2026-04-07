"""
OpenEnv compliance validation tests.

Validates:
- All required endpoints present (reset, step, state, health)
- Response schemas match OpenEnv spec
- openenv.yaml is valid
"""

import json
import urllib.request
import urllib.error
import yaml
import pytest
from pathlib import Path


class TestOpenEnvCompliance:
    """Test OpenEnv specification compliance."""

    @pytest.fixture(scope="class")
    def base_url(self):
        """Base URL for the environment API."""
        return "http://localhost:7860"

    @pytest.fixture(scope="class")
    def openenv_spec(self):
        """Load and parse openenv.yaml specification."""
        spec_path = Path("openenv-support-env/openenv.yaml")
        
        if not spec_path.exists():
            pytest.fail(f"openenv.yaml not found at {spec_path}")
        
        with open(spec_path, "r") as f:
            spec = yaml.safe_load(f)
        
        return spec

    def _make_request(self, url, method="GET", data=None):
        """Helper to make HTTP requests."""
        headers = {}
        body = None
        
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                response_data = json.loads(response.read().decode())
                return response.status, response_data
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode()) if e.headers.get("Content-Type") == "application/json" else {}
        except Exception as e:
            pytest.fail(f"Request failed: {e}")

    def test_openenv_yaml_is_valid(self, openenv_spec):
        """Verify openenv.yaml is valid and contains required fields."""
        # Check required top-level fields
        required_fields = ["name", "version", "description", "author", "tags", "tasks", "api"]
        for field in required_fields:
            assert field in openenv_spec, f"openenv.yaml missing required field: {field}"
        
        # Verify name
        assert openenv_spec["name"] == "customer-support-triage", \
            f"Expected name 'customer-support-triage', got '{openenv_spec['name']}'"
        
        # Verify version
        assert openenv_spec["version"] == "1.0.0", \
            f"Expected version '1.0.0', got '{openenv_spec['version']}'"
        
        # Verify tasks
        tasks = openenv_spec["tasks"]
        assert isinstance(tasks, list), "tasks should be a list"
        assert len(tasks) == 3, f"Expected 3 tasks, got {len(tasks)}"
        
        task_names = [task["name"] for task in tasks]
        assert "easy" in task_names, "Missing 'easy' task"
        assert "medium" in task_names, "Missing 'medium' task"
        assert "hard" in task_names, "Missing 'hard' task"
        
        # Verify each task has required fields
        for task in tasks:
            assert "name" in task, "Task missing 'name' field"
            assert "description" in task, "Task missing 'description' field"
            assert "max_steps" in task, "Task missing 'max_steps' field"
            assert "reward_range" in task, "Task missing 'reward_range' field"
            assert task["max_steps"] == 3, f"Expected max_steps=3, got {task['max_steps']}"
            assert task["reward_range"] == [0.0, 1.0], \
                f"Expected reward_range [0.0, 1.0], got {task['reward_range']}"
        
        # Verify API endpoints
        api = openenv_spec["api"]
        assert "reset" in api, "API missing 'reset' endpoint"
        assert "step" in api, "API missing 'step' endpoint"
        assert "state" in api, "API missing 'state' endpoint"
        assert "health" in api, "API missing 'health' endpoint"
        
        assert api["reset"] == "POST /reset", f"Expected 'POST /reset', got '{api['reset']}'"
        assert api["step"] == "POST /step", f"Expected 'POST /step', got '{api['step']}'"
        assert api["state"] == "GET /state", f"Expected 'GET /state', got '{api['state']}'"
        assert api["health"] == "GET /health", f"Expected 'GET /health', got '{api['health']}'"
        
        print("[PASS] openenv.yaml is valid and contains all required fields")

    def test_health_endpoint_present(self, base_url):
        """Verify GET /health endpoint is present and responds correctly."""
        status, data = self._make_request(f"{base_url}/health", method="GET")
        
        assert status == 200, f"Expected status 200, got {status}"
        assert "status" in data, "Health response missing 'status' field"
        assert "task" in data, "Health response missing 'task' field"
        assert "version" in data, "Health response missing 'version' field"
        
        assert data["status"] == "healthy", f"Expected status 'healthy', got '{data['status']}'"
        assert data["version"] == "1.0.0", f"Expected version '1.0.0', got '{data['version']}'"
        
        print(f"[PASS] GET /health endpoint present and valid: {data}")

    def test_reset_endpoint_present(self, base_url):
        """Verify POST /reset endpoint is present and responds correctly."""
        status, data = self._make_request(f"{base_url}/reset", method="POST", data={})
        
        assert status == 200, f"Expected status 200, got {status}"
        
        # Verify observation schema
        required_fields = [
            "ticket_id", "subject", "body", "customer_history",
            "available_categories", "available_priorities",
            "step_feedback", "reward", "done", "metadata"
        ]
        
        for field in required_fields:
            assert field in data, f"Reset response missing required field: {field}"
        
        # Verify field types
        assert isinstance(data["ticket_id"], str), "ticket_id should be string"
        assert isinstance(data["subject"], str), "subject should be string"
        assert isinstance(data["body"], str), "body should be string"
        assert isinstance(data["customer_history"], list), "customer_history should be list"
        assert isinstance(data["available_categories"], list), "available_categories should be list"
        assert isinstance(data["available_priorities"], list), "available_priorities should be list"
        assert isinstance(data["step_feedback"], str), "step_feedback should be string"
        assert isinstance(data["reward"], (int, float)), "reward should be number"
        assert isinstance(data["done"], bool), "done should be boolean"
        assert isinstance(data["metadata"], dict), "metadata should be dict"
        
        # Verify available categories and priorities
        expected_categories = ["billing", "technical", "account", "shipping", "general"]
        assert data["available_categories"] == expected_categories, \
            f"Expected categories {expected_categories}, got {data['available_categories']}"
        
        expected_priorities = ["low", "medium", "high", "urgent"]
        assert data["available_priorities"] == expected_priorities, \
            f"Expected priorities {expected_priorities}, got {data['available_priorities']}"
        
        # Verify initial state
        assert data["done"] is False, "Initial observation should have done=False"
        assert data["ticket_id"] != "", "Initial observation should have non-empty ticket_id"
        
        print(f"[PASS] POST /reset endpoint present and returns valid observation")

    def test_step_endpoint_present(self, base_url):
        """Verify POST /step endpoint is present and responds correctly."""
        # First reset to ensure we have an active episode
        self._make_request(f"{base_url}/reset", method="POST", data={})
        
        # Now make a step request
        action = {
            "category": "billing",
            "priority": "high",
            "response": "Thank you for contacting us. We will investigate this issue and get back to you shortly.",
            "escalate": False,
            "tags": ["billing"],
            "metadata": {}
        }
        
        status, data = self._make_request(f"{base_url}/step", method="POST", data=action)
        
        assert status == 200, f"Expected status 200, got {status}"
        
        # Verify step response schema
        required_fields = ["observation", "reward", "done", "info"]
        for field in required_fields:
            assert field in data, f"Step response missing required field: {field}"
        
        # Verify observation schema (same as reset)
        obs = data["observation"]
        obs_required_fields = [
            "ticket_id", "subject", "body", "customer_history",
            "available_categories", "available_priorities",
            "step_feedback", "reward", "done", "metadata"
        ]
        
        for field in obs_required_fields:
            assert field in obs, f"Observation missing required field: {field}"
        
        # Verify reward and done
        assert isinstance(data["reward"], (int, float)), "reward should be number"
        assert 0.0 <= data["reward"] <= 1.0, f"reward should be in [0.0, 1.0], got {data['reward']}"
        assert isinstance(data["done"], bool), "done should be boolean"
        
        # Verify info contains breakdown
        assert isinstance(data["info"], dict), "info should be dict"
        assert "breakdown" in data["info"], "info missing 'breakdown' field"
        assert "ticket_id" in data["info"], "info missing 'ticket_id' field"
        
        print(f"[PASS] POST /step endpoint present and returns valid step response")

    def test_state_endpoint_present(self, base_url):
        """Verify GET /state endpoint is present and responds correctly."""
        # First reset to ensure we have an active episode
        self._make_request(f"{base_url}/reset", method="POST", data={})
        
        # Now get state
        status, data = self._make_request(f"{base_url}/state", method="GET")
        
        assert status == 200, f"Expected status 200, got {status}"
        
        # Verify state schema
        required_fields = [
            "episode_id", "task_name", "step_count", "total_reward",
            "tickets_processed", "correct_categories", "correct_priorities",
            "escalations_made", "escalations_needed", "metadata"
        ]
        
        for field in required_fields:
            assert field in data, f"State response missing required field: {field}"
        
        # Verify field types
        assert isinstance(data["episode_id"], str), "episode_id should be string"
        assert isinstance(data["task_name"], str), "task_name should be string"
        assert isinstance(data["step_count"], int), "step_count should be int"
        assert isinstance(data["total_reward"], (int, float)), "total_reward should be number"
        assert isinstance(data["tickets_processed"], int), "tickets_processed should be int"
        assert isinstance(data["correct_categories"], int), "correct_categories should be int"
        assert isinstance(data["correct_priorities"], int), "correct_priorities should be int"
        assert isinstance(data["escalations_made"], int), "escalations_made should be int"
        assert isinstance(data["escalations_needed"], int), "escalations_needed should be int"
        assert isinstance(data["metadata"], dict), "metadata should be dict"
        
        # Verify task_name is valid
        assert data["task_name"] in ["easy", "medium", "hard"], \
            f"task_name should be easy/medium/hard, got '{data['task_name']}'"
        
        print(f"[PASS] GET /state endpoint present and returns valid state")

    def test_all_required_endpoints_present(self, base_url):
        """Verify all required endpoints (reset, step, state, health) are present."""
        endpoints = [
            ("GET", "/health"),
            ("POST", "/reset"),
            ("POST", "/step"),
            ("GET", "/state")
        ]
        
        # Reset first to enable step and state
        self._make_request(f"{base_url}/reset", method="POST", data={})
        
        for method, path in endpoints:
            url = f"{base_url}{path}"
            
            # Prepare data for POST requests
            data = None
            if method == "POST" and path == "/step":
                data = {
                    "category": "general",
                    "priority": "medium",
                    "response": "Thank you for your message. We will review your request and respond shortly.",
                    "escalate": False,
                    "tags": [],
                    "metadata": {}
                }
            elif method == "POST" and path == "/reset":
                data = {}
            
            status, response = self._make_request(url, method=method, data=data)
            
            assert status == 200, f"{method} {path} returned status {status}, expected 200"
            print(f"[PASS] {method} {path} is present and accessible")

    def test_response_schemas_match_openenv_spec(self, base_url):
        """Verify response schemas match OpenEnv specification."""
        # This is a comprehensive test that verifies all schemas together
        
        # 1. Reset and verify observation schema
        status, reset_data = self._make_request(f"{base_url}/reset", method="POST", data={})
        assert status == 200
        
        # Verify observation has all required fields with correct types
        assert isinstance(reset_data["ticket_id"], str)
        assert isinstance(reset_data["subject"], str)
        assert isinstance(reset_data["body"], str)
        assert isinstance(reset_data["customer_history"], list)
        assert isinstance(reset_data["available_categories"], list)
        assert isinstance(reset_data["available_priorities"], list)
        assert isinstance(reset_data["step_feedback"], str)
        assert isinstance(reset_data["reward"], (int, float))
        assert isinstance(reset_data["done"], bool)
        assert isinstance(reset_data["metadata"], dict)
        
        # 2. Step and verify step response schema
        action = {
            "category": "billing",
            "priority": "high",
            "response": "We apologize for the inconvenience. We will investigate this matter immediately.",
            "escalate": False,
            "tags": ["billing"],
            "metadata": {}
        }
        
        status, step_data = self._make_request(f"{base_url}/step", method="POST", data=action)
        assert status == 200
        
        # Verify step response structure
        assert "observation" in step_data
        assert "reward" in step_data
        assert "done" in step_data
        assert "info" in step_data
        
        assert isinstance(step_data["observation"], dict)
        assert isinstance(step_data["reward"], (int, float))
        assert isinstance(step_data["done"], bool)
        assert isinstance(step_data["info"], dict)
        
        # 3. State and verify state schema
        status, state_data = self._make_request(f"{base_url}/state", method="GET")
        assert status == 200
        
        # Verify state has all required fields with correct types
        assert isinstance(state_data["episode_id"], str)
        assert isinstance(state_data["task_name"], str)
        assert isinstance(state_data["step_count"], int)
        assert isinstance(state_data["total_reward"], (int, float))
        assert isinstance(state_data["tickets_processed"], int)
        assert isinstance(state_data["correct_categories"], int)
        assert isinstance(state_data["correct_priorities"], int)
        assert isinstance(state_data["escalations_made"], int)
        assert isinstance(state_data["escalations_needed"], int)
        assert isinstance(state_data["metadata"], dict)
        
        # 4. Health and verify health schema
        status, health_data = self._make_request(f"{base_url}/health", method="GET")
        assert status == 200
        
        assert isinstance(health_data["status"], str)
        assert isinstance(health_data["task"], str)
        assert isinstance(health_data["version"], str)
        
        print("[PASS] All response schemas match OpenEnv specification")
