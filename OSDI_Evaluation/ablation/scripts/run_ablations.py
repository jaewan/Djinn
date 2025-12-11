#!/usr/bin/env python3
"""
Master script to run all ablation studies sequentially.

Usage:
  Terminal 1: python3 -m djinn.server --port 5556 --gpu 0
  Terminal 2: python3 run_ablations.py
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add Djinn to path
sys.path.insert(0, '/home/ubuntu/Djinn')

# Import ablation modules
from ablation_os_tax import run_ablation as run_os_tax
from ablation_plan_cache import run_ablation as run_plan_cache


async def run_all_ablations(
    server: str = "127.0.0.1:5556",
    output_dir: Path = None,
    skip_os_tax: bool = False,
    skip_plan_cache: bool = False,
    model_sweep: bool = False
):
    """Run all ablation studies sequentially."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("RUNNING ALL ABLATION STUDIES")
    print("="*70)
    print(f"\nServer: {server}")
    print(f"Output directory: {output_dir}")
    print(f"\nAblations to run:")
    if not skip_os_tax:
        if model_sweep:
            print("  1. OS Tax (Interposition Overhead) - MODEL SWEEP")
            print("     - GPT-2 (124M)")
            print("     - Phi-2 (2.7B)")
            print("     - Llama-2-7B")
        else:
            print("  1. OS Tax (Interposition Overhead)")
    if not skip_plan_cache:
        print("  2. Plan Cache Effectiveness")
    
    results = {}
    start_time = time.time()
    
    # Ablation 1: OS Tax (with optional model sweep)
    if not skip_os_tax:
        if model_sweep:
            # Run model sweep: GPT-2, Phi-2, Llama-7B
            models_to_test = [
                ('gpt2', 'GPT-2 (124M)'),
                ('microsoft/phi-2', 'Phi-2 (2.7B)'),
                ('meta-llama/Llama-2-7b-hf', 'Llama-2-7B')
            ]
            
            os_tax_sweep_results = {}
            total_sweep_time = 0
            
            for model_id, model_name in models_to_test:
                print("\n" + "="*70)
                print(f"STARTING ABLATION 1: OS TAX - {model_name}")
                print("="*70)
                try:
                    ablation_start = time.time()
                    model_results = await run_os_tax(
                        server=server,
                        n_warmup=100,
                        n_iters=1000,
                        output_dir=output_dir,
                        model_id=model_id,
                        model_name=model_name
                    )
                    ablation_time = time.time() - ablation_start
                    total_sweep_time += ablation_time
                    
                    os_tax_sweep_results[model_id] = {
                        'status': 'success',
                        'duration_sec': ablation_time,
                        'results': model_results
                    }
                    print(f"\n✅ {model_name} completed in {ablation_time:.1f}s")
                    
                    # Wait between models to let server stabilize
                    if model_id != models_to_test[-1][0]:
                        print("\n⏳ Waiting 10 seconds before next model...")
                        await asyncio.sleep(10)
                        
                except Exception as e:
                    print(f"\n❌ {model_name} failed: {e}")
                    os_tax_sweep_results[model_id] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    import traceback
                    traceback.print_exc()
            
            results['os_tax_sweep'] = {
                'status': 'success' if any(r['status'] == 'success' for r in os_tax_sweep_results.values()) else 'failed',
                'duration_sec': total_sweep_time,
                'models': os_tax_sweep_results
            }
            print(f"\n✅ OS Tax Model Sweep completed in {total_sweep_time:.1f}s")
        else:
            # Run single model (GPT-2)
            print("\n" + "="*70)
            print("STARTING ABLATION 1: OS TAX")
            print("="*70)
            try:
                ablation_start = time.time()
                os_tax_results = await run_os_tax(
                    server=server,
                    n_warmup=100,
                    n_iters=1000,
                    output_dir=output_dir,
                    model_id='gpt2',
                    model_name='GPT-2'
                )
                ablation_time = time.time() - ablation_start
                results['os_tax'] = {
                    'status': 'success',
                    'duration_sec': ablation_time,
                    'results': os_tax_results
                }
                print(f"\n✅ Ablation 1 completed in {ablation_time:.1f}s")
            except Exception as e:
                print(f"\n❌ Ablation 1 failed: {e}")
                results['os_tax'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                import traceback
                traceback.print_exc()
    
    # Wait between ablations to let server stabilize
    if not skip_os_tax and not skip_plan_cache:
        print("\n⏳ Waiting 5 seconds before next ablation...")
        await asyncio.sleep(5)
    
    # Ablation 2: Plan Cache
    if not skip_plan_cache:
        print("\n" + "="*70)
        print("STARTING ABLATION 2: PLAN CACHE")
        print("="*70)
        try:
            ablation_start = time.time()
            plan_cache_results = await run_plan_cache(
                server=server,
                n_uniform=10,
                n_varied=10,
                n_trials=3,
                output_dir=output_dir
            )
            ablation_time = time.time() - ablation_start
            results['plan_cache'] = {
                'status': 'success',
                'duration_sec': ablation_time,
                'results': plan_cache_results
            }
            print(f"\n✅ Ablation 2 completed in {ablation_time:.1f}s")
        except Exception as e:
            print(f"\n❌ Ablation 2 failed: {e}")
            results['plan_cache'] = {
                'status': 'failed',
                'error': str(e)
            }
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"\nResults:")
    
    for ablation_name, result in results.items():
        status = result['status']
        if status == 'success':
            duration = result['duration_sec']
            print(f"  {ablation_name}: ✅ SUCCESS ({duration:.1f}s)")
        else:
            error = result.get('error', 'Unknown error')
            print(f"  {ablation_name}: ❌ FAILED - {error}")
    
    # Save summary
    summary = {
        'timestamp': time.time(),
        'server': server,
        'total_duration_sec': total_time,
        'ablations': results
    }
    
    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary saved to: {summary_path}")
    
    # Print file locations
    print(f"\n📁 Output files:")
    print(f"  Results directory: {output_dir}")
    if not skip_os_tax:
        print(f"  OS Tax: {output_dir / 'ablation_os_tax.json'}")
        print(f"  OS Tax LaTeX: {output_dir / 'os_tax_table.tex'}")
    if not skip_plan_cache:
        print(f"  Plan Cache: {output_dir / 'ablation_plan_cache.json'}")
        print(f"  Plan Cache LaTeX: {output_dir / 'plan_cache_table.tex'}")
    print(f"  Summary: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run all ablation studies")
    parser.add_argument('--server', default='127.0.0.1:5556', help='Djinn server address')
    parser.add_argument('--output', type=Path, default=None, help='Output directory')
    parser.add_argument('--skip-os-tax', action='store_true', help='Skip OS Tax ablation')
    parser.add_argument('--skip-plan-cache', action='store_true', help='Skip Plan Cache ablation')
    parser.add_argument('--model-sweep', action='store_true', help='Run OS Tax with model sweep (GPT-2, Phi-2, Llama-7B)')
    args = parser.parse_args()
    
    if args.skip_os_tax and args.skip_plan_cache:
        print("❌ Error: Cannot skip all ablations")
        sys.exit(1)
    
    asyncio.run(run_all_ablations(
        server=args.server,
        output_dir=args.output,
        skip_os_tax=args.skip_os_tax,
        skip_plan_cache=args.skip_plan_cache,
        model_sweep=args.model_sweep
    ))


if __name__ == '__main__':
    main()
