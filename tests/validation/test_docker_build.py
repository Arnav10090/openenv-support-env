"""
Docker build validation tests.

Validates:
- Docker image builds successfully without errors
- Container starts and responds to /health within 10 seconds
- Port 7860 is accessible
"""

import subprocess
import time
import urllib.request
import urllib.error
import json
import pytest


class TestDockerBuild:
    """Test Docker build and container startup."""

    @pytest.fixture(scope="class")
    def docker_image_name(self):
        """Generate unique image name for this test run."""
        return f"support-env-test:{int(time.time())}"

    @pytest.fixture(scope="class")
    def build_docker_image(self, docker_image_name):
        """Build Docker image and return image name."""
        print(f"\n[BUILD] Building Docker image: {docker_image_name}")
        
        # Build the Docker image
        result = subprocess.run(
            ["docker", "build", "-t", docker_image_name, "."],
            cwd="openenv-support-env",
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for build
        )
        
        if result.returncode != 0:
            print(f"[BUILD ERROR] stdout: {result.stdout}")
            print(f"[BUILD ERROR] stderr: {result.stderr}")
            pytest.fail(f"Docker build failed with exit code {result.returncode}")
        
        print(f"[BUILD] Successfully built image: {docker_image_name}")
        yield docker_image_name
        
        # Cleanup: remove the image after tests
        print(f"\n[CLEANUP] Removing Docker image: {docker_image_name}")
        subprocess.run(
            ["docker", "rmi", "-f", docker_image_name],
            capture_output=True
        )

    @pytest.fixture(scope="class")
    def running_container(self, build_docker_image):
        """Start container and return container ID."""
        image_name = build_docker_image
        print(f"\n[CONTAINER] Starting container from image: {image_name}")
        
        # Start the container
        result = subprocess.run(
            ["docker", "run", "-d", "-p", "7860:7860", image_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"[CONTAINER ERROR] stdout: {result.stdout}")
            print(f"[CONTAINER ERROR] stderr: {result.stderr}")
            pytest.fail(f"Failed to start container with exit code {result.returncode}")
        
        container_id = result.stdout.strip()
        print(f"[CONTAINER] Started container: {container_id}")
        
        # Wait a moment for container to initialize
        time.sleep(2)
        
        yield container_id
        
        # Cleanup: stop and remove the container
        print(f"\n[CLEANUP] Stopping container: {container_id}")
        subprocess.run(["docker", "stop", container_id], capture_output=True)
        subprocess.run(["docker", "rm", container_id], capture_output=True)

    def test_docker_build_succeeds(self, build_docker_image):
        """Verify docker build . succeeds without errors."""
        # If we reach here, the build_docker_image fixture succeeded
        assert build_docker_image is not None
        print(f"[PASS] Docker build succeeded for image: {build_docker_image}")

    def test_container_starts_and_health_responds(self, running_container):
        """Verify container starts and responds to /health within 10 seconds."""
        container_id = running_container
        health_url = "http://localhost:7860/health"
        
        print(f"[HEALTH] Waiting for health endpoint at {health_url}")
        
        # Try to connect to health endpoint within 10 seconds
        start_time = time.time()
        max_wait = 10
        connected = False
        last_error = None
        
        while time.time() - start_time < max_wait:
            try:
                req = urllib.request.Request(health_url, method="GET")
                with urllib.request.urlopen(req, timeout=2) as response:
                    data = json.loads(response.read().decode())
                    print(f"[HEALTH] Response: {data}")
                    
                    # Verify response structure
                    assert "status" in data, "Health response missing 'status' field"
                    assert data["status"] == "healthy", f"Expected status 'healthy', got '{data['status']}'"
                    assert "task" in data, "Health response missing 'task' field"
                    assert "version" in data, "Health response missing 'version' field"
                    
                    connected = True
                    elapsed = time.time() - start_time
                    print(f"[PASS] Health endpoint responded in {elapsed:.2f} seconds")
                    break
                    
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError) as e:
                last_error = e
                time.sleep(0.5)
        
        if not connected:
            # Check container logs for debugging
            logs_result = subprocess.run(
                ["docker", "logs", container_id],
                capture_output=True,
                text=True
            )
            print(f"[CONTAINER LOGS]\n{logs_result.stdout}")
            if logs_result.stderr:
                print(f"[CONTAINER STDERR]\n{logs_result.stderr}")
            
            pytest.fail(
                f"Container did not respond to /health within {max_wait} seconds. "
                f"Last error: {last_error}"
            )

    def test_port_7860_accessible(self, running_container):
        """Verify port 7860 is accessible."""
        # This test verifies we can make a successful HTTP request to port 7860
        health_url = "http://localhost:7860/health"
        
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                assert response.status == 200, f"Expected status 200, got {response.status}"
                print(f"[PASS] Port 7860 is accessible and responding")
        except Exception as e:
            pytest.fail(f"Port 7860 is not accessible: {e}")
