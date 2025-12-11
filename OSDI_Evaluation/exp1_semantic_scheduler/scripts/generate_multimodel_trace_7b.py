#!/usr/bin/env python3
"""
Generate multi-model traces for 7B-class models (OSDI experiments).

Creates traces with agents distributed across Llama-7B, Mistral-7B, and Phi-2.
"""

import json
import logging
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_prompt(task_type: str, length: int = 512) -> str:
    """Generate a prompt based on task type."""
    prompts = {
        "complex": "Analyze the following complex scenario and provide a detailed reasoning: ",
        "medium": "Explain the following concept in detail: ",
        "simple": "Answer the following question briefly: "
    }
    
    base = prompts.get(task_type, prompts["medium"])
    
    # Add filler to reach desired length (approximate)
    filler = "The quick brown fox jumps over the lazy dog. " * (length // 50)
    return base + filler[:length]


def generate_multimodel_trace(
    n_agents: int,
    model_configs: Dict[str, Dict],
    model_distribution: Dict[str, float],
    arrival_rate: float,
    think_time_range: List[float],
    context_length: int,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate a multi-model trace with Poisson arrivals.
    
    Args:
        n_agents: Number of agents
        model_configs: Model configuration dict
        model_distribution: Distribution of agents across models
        arrival_rate: Agents per second
        think_time_range: [min, max] think time in seconds
        context_length: Input context length
        seed: Random seed
        
    Returns:
        Trace dictionary
    """
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Generating trace: {n_agents} agents, rate={arrival_rate}/s")
    
    # Generate Poisson arrival times
    inter_arrival_times = np.random.exponential(1.0 / arrival_rate, n_agents)
    arrival_times = np.cumsum(inter_arrival_times)
    
    # Assign models based on distribution
    models = list(model_distribution.keys())
    weights = [model_distribution[m] for m in models]
    
    agents = []
    for i in range(n_agents):
        # Select model
        model = np.random.choice(models, p=weights)
        task_type = model_configs[model]['task_type']
        
        # Generate prompt
        prompt = generate_prompt(task_type, context_length)
        
        # Generate think time
        think_time = random.uniform(think_time_range[0], think_time_range[1])
        
        agents.append({
            "agent_id": i,
            "model": model,
            "task_type": task_type,
            "arrival_time": float(arrival_times[i]),
            "think_time": float(think_time),
            "prompt_text": prompt,
        })
    
    # Count agents per model
    model_counts = {}
    for agent in agents:
        model = agent['model']
        model_counts[model] = model_counts.get(model, 0) + 1
    
    logger.info(f"Agent distribution:")
    for model, count in model_counts.items():
        logger.info(f"  {model}: {count} ({count/n_agents*100:.1f}%)")
    
    return {
        "n_agents": n_agents,
        "arrival_rate": arrival_rate,
        "context_length": context_length,
        "think_time_range": think_time_range,
        "seed": seed,
        "agents": agents,
        "model_configs": model_configs,
        "model_distribution": model_distribution,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate multi-model trace for 7B experiments")
    parser.add_argument("--config", type=str, default="../configs/multimodel_7b.yaml",
                        help="Config file with model specifications")
    parser.add_argument("--n-agents", type=int, default=50,
                        help="Number of agents")
    parser.add_argument("--output", type=str, default="../traces/multimodel_7b_n50.json",
                        help="Output trace file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Generate trace
    trace = generate_multimodel_trace(
        n_agents=args.n_agents,
        model_configs=config['models'],
        model_distribution=config['experiment']['model_distribution'],
        arrival_rate=config['experiment']['arrival_rate'],
        think_time_range=config['experiment']['think_time_range'],
        context_length=config['experiment']['context_length'],
        seed=args.seed,
    )
    
    # Save trace
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(trace, f, indent=2)
    
    logger.info(f"✅ Trace saved: {output_path}")
    logger.info(f"   Total agents: {trace['n_agents']}")
    logger.info(f"   Duration: {trace['agents'][-1]['arrival_time']:.1f}s")


if __name__ == "__main__":
    main()
