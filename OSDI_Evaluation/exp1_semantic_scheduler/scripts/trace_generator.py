#!/usr/bin/env python3
"""
Trace Generator for OSDI Baselines.

Generates deterministic, reproducible workload traces that all baselines
(vLLM, Ray, Serverless, Djinn) will use for fair comparison.

Each trace contains:
- Poisson arrival times
- Fixed prompts (2048 tokens)
- Think times (I/O wait simulation)
- Random seed for reproducibility
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timezone

import numpy as np
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_trace(
    n_agents: int,
    arrival_rate: float,
    think_time_min: float,
    think_time_max: float,
    context_length: int,
    model_id: str = "meta-llama/Llama-2-13b-hf",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a deterministic workload trace.

    Args:
        n_agents: Number of concurrent agents
        arrival_rate: Poisson arrival rate (agents/sec)
        think_time_min: Minimum think time (sec)
        think_time_max: Maximum think time (sec)
        context_length: Input context length (tokens)
        model_id: Model ID for tokenization
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing trace data
    """
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Generating trace for N={n_agents}, arrival_rate={arrival_rate}")

    # Generate Poisson arrival times
    arrival_times = []
    current_time = 0.0

    for _ in range(n_agents):
        inter_arrival = np.random.exponential(1.0 / arrival_rate)
        current_time += inter_arrival
        arrival_times.append(current_time)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create a single prompt repeated N times (same for all agents)
    base_prompt = """
    We the People of the United States, in Order to form a more perfect Union,
    establish Justice, insure domestic Tranquility, provide for the common defence,
    promote the general Welfare, and secure the Blessings of Liberty to ourselves
    and our Posterity, do ordain and establish this Constitution for the United States of America.
    Article I: The Legislative Branch. Congress shall have Power To lay and collect Taxes,
    Duties, Imposts and Excises, to pay the Debts and provide for the common Defence and general Welfare.
    Section 1. All legislative Powers herein granted shall be vested in a Congress of the United States,
    which shall consist of a Senate and House of Representatives.
    Section 2. The House of Representatives shall be composed of Members chosen every second Year
    by the People of the several States, and the Electors in each State shall have the Qualifications
    requisite for Electors of the most numerous Branch of the State Legislature.
    """

    # Tokenize to get exact context_length
    tokens = tokenizer.encode(base_prompt)[:context_length]
    prompt_text = tokenizer.decode(tokens)

    logger.info(f"Prompt: {len(tokens)} tokens")

    # Create identical prompts for all agents
    prompts = [prompt_text] * n_agents

    # Generate think times (simulated I/O wait)
    think_times = [
        random.uniform(think_time_min, think_time_max) for _ in range(n_agents)
    ]

    trace = {
        "n_agents": n_agents,
        "arrival_rate": arrival_rate,
        "think_time_min": think_time_min,
        "think_time_max": think_time_max,
        "context_length": context_length,
        "model_id": model_id,
        "seed": seed,
        "arrival_times": arrival_times,  # Wall-clock seconds from start
        "prompts": prompts,  # Text prompts (identical for all)
        "prompt_tokens": tokens,  # Tokenized (for efficiency)
        "think_times": think_times,  # Simulated I/O wait per agent
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    return trace


def save_trace(trace: Dict[str, Any], output_path: Path) -> None:
    """Save trace to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trace, f, indent=2)
    logger.info(f"Saved trace to {output_path}")


def load_trace(trace_path: Path) -> Dict[str, Any]:
    """Load trace from JSON file."""
    with open(trace_path) as f:
        return json.load(f)


def main():
    """Generate traces for all N values."""
    # Use absolute path relative to this script
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "traces"
    output_dir.mkdir(parents=True, exist_ok=True)

    agent_counts = [10, 20, 30, 40, 50, 60, 70, 80]

    for n in agent_counts:
        trace = generate_trace(
            n_agents=n,
            arrival_rate=0.2,  # 1 agent per 5 seconds
            think_time_min=10.0,
            think_time_max=20.0,
            context_length=2048,
            model_id="meta-llama/Llama-2-13b-hf",
            seed=42,  # Fixed seed for reproducibility
        )

        output_path = output_dir / f"trace_{n}.json"
        save_trace(trace, output_path)

    logger.info(f"Generated traces for N={agent_counts}")


if __name__ == "__main__":
    main()
