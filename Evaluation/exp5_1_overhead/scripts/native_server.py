#!/usr/bin/env python3
"""
Native PyTorch server for fair baseline comparison.

This server runs model.generate() in a dedicated process, matching
the process isolation of RPC and Djinn servers for apples-to-apples comparison.

Usage:
    python native_server.py --port 5557 --device cuda:0 \
        --config configs/overhead_hf_smoke.yaml --workloads hf_tiny_gpt2
"""

import argparse
import json
import socket
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

# Add repo root to path
import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Evaluation.common.workloads import build_workload


class NativeServer:
    """Simple socket server for native PyTorch baseline."""
    
    def __init__(self, port: int, device: str, dtype: str):
        self.port = port
        self.device = torch.device(device)
        self.dtype = dtype
        self.workloads: Dict[str, Any] = {}
        self.sock = None
        
    def load_workloads(self, config_path: str, only: Optional[set] = None):
        """Load workloads from config file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        for wl_cfg in cfg.get("workloads", []):
            name = wl_cfg["name"]
            if only and name not in only:
                continue
            
            impl = wl_cfg["implementation"]
            spec = wl_cfg.get("params", {})
            
            print(f"[native_server] Loading workload '{name}'...")
            workload = build_workload(impl, spec, str(self.device), self.dtype)
            self.workloads[name] = {
                "workload": workload,
                "implementation": impl,
            }
            print(f"[native_server] Loaded workload '{name}' on {self.device}")
        
        # Warmup CUDA
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            print("[native_server] CUDA context initialized")
    
    def execute(self, workload_name: str, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute workload with same semantics as other baselines."""
        if workload_name not in self.workloads:
            raise ValueError(f"Unknown workload: {workload_name}")
        
        wl_info = self.workloads[workload_name]
        workload = wl_info["workload"]
        impl = wl_info["implementation"]
        model = workload.model
        
        # Move inputs to device
        gpu_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if key in {"input_ids", "attention_mask", "token_type_ids"}:
                    gpu_inputs[key] = value.to(self.device)
                else:
                    gpu_inputs[key] = value.to(self.device, dtype=model.dtype)
            else:
                gpu_inputs[key] = value
        
        # Execute with same context as other baselines
        with torch.no_grad():
            if impl == "hf_causal_lm":
                gen_kwargs = dict(workload.generation_params)
                gen_kwargs["max_new_tokens"] = workload.new_tokens
                output = model.generate(**gpu_inputs, **gen_kwargs)
            elif impl.startswith("synthetic"):
                output = model(gpu_inputs.get("x"))
            else:
                output = model(**gpu_inputs)
                if hasattr(output, "logits"):
                    output = output.logits
            
            # Ensure GPU work is complete
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        
        return output.detach().cpu()
    
    def serve(self):
        """Start server and handle requests."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.listen(5)
        
        print(f"[native_server] Listening on port {self.port}")
        
        while True:
            conn, addr = self.sock.accept()
            try:
                self._handle_connection(conn)
            except Exception as e:
                print(f"[native_server] Error handling connection: {e}")
            finally:
                conn.close()
    
    def _handle_connection(self, conn):
        """Handle a single client connection."""
        # Read request length
        length_bytes = conn.recv(4)
        if not length_bytes:
            return
        
        msg_len = struct.unpack(">I", length_bytes)[0]
        
        # Read request data
        data = b""
        while len(data) < msg_len:
            chunk = conn.recv(min(msg_len - len(data), 65536))
            if not chunk:
                break
            data += chunk
        
        # Parse request
        request = json.loads(data.decode())
        workload_name = request["workload"]
        inputs = {}
        for key, val in request["inputs"].items():
            if isinstance(val, list):
                inputs[key] = torch.tensor(val)
            else:
                inputs[key] = val
        
        # Execute
        start = time.perf_counter()
        output = self.execute(workload_name, inputs)
        latency_ms = (time.perf_counter() - start) * 1000.0
        
        # Send response
        response = {
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "latency_ms": latency_ms,
        }
        response_bytes = json.dumps(response).encode()
        conn.sendall(struct.pack(">I", len(response_bytes)))
        conn.sendall(response_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workloads", nargs="*")
    args = parser.parse_args()
    
    server = NativeServer(args.port, args.device, args.dtype)
    
    only = set(args.workloads) if args.workloads else None
    server.load_workloads(args.config, only)
    
    server.serve()


if __name__ == "__main__":
    main()

