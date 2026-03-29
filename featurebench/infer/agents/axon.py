"""
Axon agent implementation — axon_ex API mode.

Submits tasks to the axon_ex Phoenix server (localhost:50051) which
runs the full Vanguard 8-gate workflow with eval loop. The testbed
is copied to the host so axon_ex's MissionRunner can access it.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import requests
from docker.models.containers import Container

from featurebench.infer.agents.base import BaseAgent


_WORKDIR = "/data3/tmp/featurebench-testbeds"
_AXON_EX_URL = "http://localhost:50051"
_POLL_INTERVAL = 10  # seconds


class AxonAgent(BaseAgent):
    """Axon agent that delegates to axon_ex server via HTTP API."""

    @property
    def name(self) -> str:
        return "axon"

    def get_extra_volumes(self) -> Dict[str, Dict[str, str]]:
        return {}

    @property
    def install_script(self) -> str:
        return '#!/bin/bash\necho "Axon runs via axon_ex API — no container install needed."'

    def get_env_setup_script(self) -> str:
        return "#!/bin/bash\n"

    def get_run_command(self, instruction: str) -> str:
        return "echo 'axon runs via axon_ex API'"

    # ------------------------------------------------------------------
    # axon_ex API helpers
    # ------------------------------------------------------------------

    def _api(self, endpoint: str, payload: dict) -> dict:
        url = f"{_AXON_EX_URL}{endpoint}"
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _poll_until_done(
        self, run_id: str, issue_id: str, log_file: Path, timeout: int
    ) -> dict:
        """Poll MissionStatus until run completes or times out."""
        deadline = time.time() + timeout
        last_gate = ""

        while time.time() < deadline:
            try:
                status = self._api("/rpc/MissionStatus", {"mission_id": run_id})
            except Exception:
                try:
                    state = self._api("/rpc/GetIssueState", {"issue_id": issue_id})
                    runs = state.get("runs", [])
                    run_info = next((r for r in runs if r.get("id") == run_id), {})
                    run_status = run_info.get("status", "unknown")
                    if run_status in ("completed", "failed", "error"):
                        return {"status": run_status}
                except Exception:
                    pass
                time.sleep(_POLL_INTERVAL)
                continue

            current_gate = status.get("current_gate", "")
            run_status = status.get("status", "unknown")

            if current_gate != last_gate:
                last_gate = current_gate
                progress = status.get("progress", 0)
                self.logger.info(
                    f"Gate: {current_gate} | Progress: {progress:.0f}%"
                )
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"Gate: {current_gate} | Progress: {progress:.0f}%\n")

            if run_status in ("completed", "failed", "error"):
                return status

            time.sleep(_POLL_INTERVAL)

        return {"status": "timeout"}

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        container: Container,
        instruction: str,
        log_file: Path,
        timeout: int | None = None,
    ) -> bool:
        self.logger.info("Running Axon via axon_ex API...")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"BEGIN Agent Execution: {self.name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Instruction:\n{instruction}\n\n")
            f.write("-" * 60 + "\n\n")

        host_dir = Path(_WORKDIR) / container.short_id
        testbed = host_dir / "testbed"
        log_dir = Path(log_file).parent
        axon_log = log_dir / "axon_output.log"

        try:
            # 1. Copy /testbed from container to host
            self.logger.info(f"Copying testbed to host: {testbed}")
            if host_dir.exists():
                shutil.rmtree(host_dir)
            host_dir.mkdir(parents=True)

            cp = subprocess.run(
                ["docker", "cp", f"{container.id}:/testbed", str(testbed)],
                capture_output=True, text=True, timeout=120,
            )
            if cp.returncode != 0:
                self.logger.error(f"docker cp out failed: {cp.stderr}")
                return False

            # 2. Prepare instruction with featurebench preamble
            testbed_str = str(testbed)
            adapted_instruction = instruction.replace("/testbed/", f"{testbed_str}/")
            adapted_instruction = instruction.replace("/testbed", testbed_str)

            preamble = (
                "## FeatureBench Coding Task\n\n"
                "This is an isolated coding benchmark. You are given a codebase and must "
                "implement the specified interfaces to pass the test suite.\n\n"
                "**Rules:**\n"
                f"- The codebase is at `{testbed_str}`. Work ONLY in this directory.\n"
                "- Read the existing code to understand patterns before writing.\n"
                "- Do NOT reference or use any prior project context — this task is self-contained.\n"
                "- Focus strictly on the interface descriptions provided below.\n"
                "- After implementation, ensure your code is syntactically correct.\n\n"
                "---\n\n"
            )

            full_body = preamble + adapted_instruction

            # Create issue in axon_ex
            model = self.env_vars.get("AXON_MODEL", "glm-5")
            issue = self._api("/rpc/CreateIssue", {
                "title": f"[featurebench] {instruction[:80]}",
                "body": full_body,
                "priority": 1,
                "workflow_preset_id": "vanguard",
            })
            issue_id = issue["id"]
            self.logger.info(f"Created issue: {issue_id}")

            with open(axon_log, "w", encoding="utf-8") as f:
                f.write(f"Issue: {json.dumps(issue, indent=2)}\n\n")

            # 3. Start run
            run_resp = self._api("/rpc/StartRun", {
                "issue_id": issue_id,
                "cwd": str(testbed),
                "model": model,
            })
            run_id = run_resp["id"]
            self.logger.info(f"Started run: {run_id}")

            with open(axon_log, "a", encoding="utf-8") as f:
                f.write(f"Run: {json.dumps(run_resp, indent=2)}\n\n")

            # 4. Poll until done
            effective_timeout = timeout or 1800
            result = self._poll_until_done(
                run_id, issue_id, log_file, effective_timeout
            )
            run_status = result.get("status", "unknown")
            self.logger.info(f"Run finished: {run_status}")

            with open(axon_log, "a", encoding="utf-8") as f:
                f.write(f"\nFinal status: {json.dumps(result, indent=2)}\n")

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\nAxon run status: {run_status}\n")

            # 5. Generate patch from host testbed and apply to container
            self.logger.info("Generating patch from host testbed...")
            diff_proc = subprocess.run(
                ["git", "diff"],
                cwd=str(testbed),
                capture_output=True, text=True, timeout=30,
            )
            patch_text = diff_proc.stdout

            if patch_text.strip():
                self.logger.info(
                    f"Patch: {len(patch_text)} chars, applying to container..."
                )
                self.cm.exec_command(
                    container,
                    f"cat > /tmp/axon.patch << 'AXONPATCHEOF'\n{patch_text}\nAXONPATCHEOF",
                    log_file=log_file,
                )
                exit_code, _ = self.cm.exec_command(
                    container,
                    "cd /testbed && git apply /tmp/axon.patch",
                    log_file=log_file,
                )
                if exit_code != 0:
                    self.logger.warning("git apply failed, trying --3way")
                    self.cm.exec_command(
                        container,
                        "cd /testbed && git apply --3way /tmp/axon.patch",
                        log_file=log_file,
                    )
            else:
                self.logger.warning("Axon produced no changes")

            return True

        except Exception as e:
            self.logger.error(f"Axon execution failed: {e}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\nERROR: {e}\n")
            return False
        finally:
            if host_dir.exists():
                shutil.rmtree(host_dir, ignore_errors=True)

            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"END Agent Execution: {self.name}\n")
                f.write("=" * 60 + "\n\n")

    def pre_run_hook(self, container, log_file) -> bool:
        return True

    def post_run_hook(self, container, log_file) -> bool:
        return True
