#!/usr/bin/env python3
"""
Sequential baseline runner for Experiment 2 (Agent Scaling).

Runs all three baselines sequentially:
1. Ray Keep-Alive (uses Ray, no Djinn server needed)
2. Ray Serverless (uses Ray, no Djinn server needed)
3. Djinn (requires Djinn server - starts/stops it)

Each baseline runs in isolation with proper process cleanup.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import signal
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def kill_process_on_port(port: int) -> None:
    """Kill any process using the specified port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], check=False)
                    print(f"[Cleanup] Killed process {pid} on port {port}")
                except Exception:
                    pass
    except FileNotFoundError:
        # lsof not available, try fuser
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            pass
    except Exception as e:
        print(f"[Warning] Could not kill process on port {port}: {e}")


def cleanup_djinn_ports() -> None:
    """Clean up all Djinn-related ports."""
    ports_to_clean = [5555, 5556, 5560] + list(range(5561, 5600))  # Control, data, and coordinator range
    for port in ports_to_clean:
        kill_process_on_port(port)


def check_server_ready(port: int, timeout: int = 30) -> bool:
    """Check if server is ready by attempting a connection."""
    import socket
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            if result == 0:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def start_djinn_server(port: int = 5556) -> subprocess.Popen:
    """Start Djinn server process."""
    print(f"[Djinn Server] Starting on port {port}...")
    
    # Clean up all Djinn-related ports
    print("[Djinn Server] Cleaning up existing processes...")
    cleanup_djinn_ports()
    
    time.sleep(2)  # Give ports time to free
    
    # Start server (capture stderr for debugging)
    server_log = Path(f"/tmp/djinn_server_{port}.log")
    with open(server_log, 'w') as log_file:
        server_process = subprocess.Popen(
            [
                sys.executable, "-m", "djinn.server.server_main",
                "--gpu", "0",
                "--port", str(port),
                "--host", "0.0.0.0",
                "--log-level", "info"
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT)
        )
    
    # Wait for server to be ready
    print(f"[Djinn Server] Waiting for server to be ready on port {port}...")
    
    # Check if process died immediately
    time.sleep(1)
    if server_process.poll() is not None:
        # Read error log
        server_log_path = Path(f"/tmp/djinn_server_{port}.log")
        error_msg = ""
        if server_log_path.exists():
            with open(server_log_path, 'r') as f:
                error_msg = f.read()[-1000:]  # Last 1000 chars
        raise RuntimeError(
            f"Djinn server process died (exit code: {server_process.returncode})\n"
            f"Server log:\n{error_msg}"
        )
    
    # Wait for port to be ready (shorter timeout since server starts quickly)
    if check_server_ready(port, timeout=10):
        print(f"[Djinn Server] Ready (PID: {server_process.pid})")
    else:
        # Check log for "Server running" message
        server_log_path = Path(f"/tmp/djinn_server_{port}.log")
        if server_log_path.exists():
            with open(server_log_path, 'r') as f:
                log_content = f.read()
                if "Server running" in log_content or "Djinn server ready" in log_content:
                    print(f"[Djinn Server] Ready (PID: {server_process.pid}) - confirmed via log")
                else:
                    print(f"[Djinn Server] Warning: Port {port} check failed, but process is alive (PID: {server_process.pid})")
                    print("[Djinn Server] Continuing anyway...")
        else:
            print(f"[Djinn Server] Warning: Port {port} check failed, but process is alive (PID: {server_process.pid})")
            print("[Djinn Server] Continuing anyway...")
    
    # Give server extra time to fully initialize
    time.sleep(2)
    
    return server_process


def stop_djinn_server(server_process: subprocess.Popen) -> None:
    """Stop Djinn server process."""
    if server_process.poll() is None:
        print(f"[Djinn Server] Stopping (PID: {server_process.pid})...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
        print("[Djinn Server] Stopped")
    else:
        print("[Djinn Server] Already stopped")


def run_ray_keepalive(args: argparse.Namespace) -> bool:
    """Run Ray Keep-Alive baseline."""
    print("\n" + "=" * 70)
    print("BASELINE 1: Ray Keep-Alive")
    print("=" * 70)
    
    cmd = [
        sys.executable,
        str(REPO_ROOT / "Evaluation/exp2_1_llm_decode/scripts/run_ray_keepalive_agents.py"),
        "--agent-counts"] + [str(n) for n in args.agent_counts] + [
        "--iterations", str(args.iterations),
        "--new-tokens", str(args.new_tokens),
        "--sleep-seconds", str(args.sleep_seconds),
        "--model-id", args.model_id,
        "--output-dir", str(args.output_dir / "ray_keepalive"),
    ]
    
    if args.ray_address:
        cmd.extend(["--ray-address", args.ray_address])
    
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    return result.returncode == 0


def run_ray_serverless(args: argparse.Namespace) -> bool:
    """Run Ray Serverless baseline."""
    print("\n" + "=" * 70)
    print("BASELINE 2: Ray Serverless")
    print("=" * 70)
    
    cmd = [
        sys.executable,
        str(REPO_ROOT / "Evaluation/exp2_1_llm_decode/scripts/run_ray_serverless_agents.py"),
        "--agent-counts"] + [str(n) for n in args.agent_counts] + [
        "--iterations", str(args.iterations),
        "--new-tokens", str(args.new_tokens),
        "--sleep-seconds", str(args.sleep_seconds),
        "--model-id", args.model_id,
        "--output-dir", str(args.output_dir / "ray_serverless"),
    ]
    
    if args.ray_address:
        cmd.extend(["--ray-address", args.ray_address])
    
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    return result.returncode == 0


def run_djinn(args: argparse.Namespace, server_process: subprocess.Popen) -> bool:
    """Run Djinn baseline."""
    print("\n" + "=" * 70)
    print("BASELINE 3: Djinn")
    print("=" * 70)
    
    cmd = [
        sys.executable,
        str(REPO_ROOT / "Evaluation/exp2_1_llm_decode/scripts/run_djinn_agents.py"),
        "--agent-counts"] + [str(n) for n in args.agent_counts] + [
        "--iterations", str(args.iterations),
        "--new-tokens", str(args.new_tokens),
        "--sleep-seconds", str(args.sleep_seconds),
        "--model-id", args.model_id,
        "--djinn-server", f"localhost:{args.djinn_port}",
        "--output-dir", str(args.output_dir / "djinn_agents"),
    ]
    
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all baselines sequentially")
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--new-tokens", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--model-id", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--output-dir", type=Path, default=Path("Evaluation/exp2_1_llm_decode/results"))
    parser.add_argument("--djinn-port", type=int, default=5556)
    parser.add_argument("--ray-address", type=str, help="Ray cluster address (optional)")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Sequential Baseline Runner for Experiment 2")
    print("=" * 70)
    print(f"Agent counts: {args.agent_counts}")
    print(f"Model: {args.model_id}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    results = {}
    djinn_server = None
    
    try:
        # Baseline 1: Ray Keep-Alive (no Djinn server needed)
        results["ray_keepalive"] = run_ray_keepalive(args)
        
        # Baseline 2: Ray Serverless (no Djinn server needed)
        results["ray_serverless"] = run_ray_serverless(args)
        
        # Baseline 3: Djinn (requires Djinn server)
        print("\n[Djinn] Starting Djinn server...")
        djinn_server = start_djinn_server(args.djinn_port)
        
        try:
            results["djinn"] = run_djinn(args, djinn_server)
        except Exception as e:
            print(f"[Djinn] Error during execution: {e}")
            results["djinn"] = False
        
    finally:
        # Always clean up Djinn server
        if djinn_server:
            stop_djinn_server(djinn_server)
        # Clean up all Djinn-related ports
        print("\n[Cleanup] Cleaning up all Djinn ports...")
        cleanup_djinn_ports()
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for baseline, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{baseline:20s}: {status}")
    print("=" * 70)
    
    if all(results.values()):
        print("\n✅ All baselines completed successfully!")
        return 0
    else:
        print("\n⚠️  Some baselines failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

