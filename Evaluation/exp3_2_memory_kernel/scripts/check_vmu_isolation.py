#!/usr/bin/env python3
"""
Session isolation sanity test for the Unified VMU.

Allocates two sessions, writes a canary pattern into session A, and verifies that
session B cannot observe the bytes even when attempting to access session A's
offset. This is intended as a quick correctness check that complements the
performance-focused harnesses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from djinn.backend.runtime.unified_vmu import UnifiedVMU  # noqa: E402

MB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--session-bytes", type=int, default=32 * MB, help="Per-session arena size in bytes")
    parser.add_argument("--canary-bytes", type=int, default=4 * MB, help="Payload size to check for isolation")
    return parser.parse_args()


def reserve_sessions(vmu: UnifiedVMU, session_bytes: int) -> Tuple[str, str]:
    session_a = "isolation_session_a"
    session_b = "isolation_session_b"
    vmu.reserve_session_arena(session_a, session_bytes)
    vmu.reserve_session_arena(session_b, session_bytes)
    return session_a, session_b


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device)
    vmu = UnifiedVMU(device_id=args.device)
    session_a, session_b = reserve_sessions(vmu, args.session_bytes)

    payload_bytes = min(args.canary_bytes, args.session_bytes // 2)
    pattern = torch.arange(payload_bytes, dtype=torch.uint8, device=vmu.device) % 251

    offset_a = vmu.allocate_session_data(session_a, payload_bytes, name="isolation_canary")
    offset_b = vmu.allocate_session_data(session_b, payload_bytes, name="isolation_control")

    view_a = vmu.get_session_data_view(session_a, offset_a, payload_bytes, torch.uint8)
    view_a.copy_(pattern)

    view_b = vmu.get_session_data_view(session_b, offset_b, payload_bytes, torch.uint8)
    view_b.zero_()

    leak_blocked = False
    try:
        _ = vmu.get_session_data_view(session_b, offset_a, payload_bytes, torch.uint8)
    except RuntimeError:
        leak_blocked = True

    if not leak_blocked:
        raise AssertionError(
            "Cross-session data view unexpectedly succeeded; session offsets should be private."
        )

    same_session_ok = torch.allclose(
        vmu.get_session_data_view(session_a, offset_a, payload_bytes, torch.uint8), pattern
    )
    isolated = torch.count_nonzero(view_b).item() == 0

    if not same_session_ok:
        raise AssertionError("Session A could not read back its own data pattern.")

    if not isolated:
        raise AssertionError("Session B view mutated unexpectedly after Session A write.")

    print("✅ VMU session isolation sanity check passed.")


if __name__ == "__main__":
    main()

