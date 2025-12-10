"""
Master runner for all OSDI ablation studies.

This script orchestrates the execution of all four ablation studies and
generates a comprehensive report with figures and tables.

Usage:
    python run_all_ablations.py [--skip-ablation N] [--output-dir DIR]
"""

import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List
import argparse

# Add Djinn to path
sys.path.insert(0, '/home/ubuntu/Djinn')


class AblationRunner:
    """Orchestrate execution of all ablation studies."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.timings = {}
    
    def run_ablation(self, ablation_num: int, script_name: str, description: str, skip: bool = False) -> bool:
        """
        Run a single ablation study.
        
        Args:
            ablation_num: Ablation number (1-4)
            script_name: Name of the script file (without .py)
            description: Human-readable description
            skip: If True, skip this ablation
        
        Returns:
            True if successful, False otherwise
        """
        if skip:
            print(f"\n⏭️  SKIPPING Ablation {ablation_num}: {description}")
            return True
        
        print(f"\n{'='*80}")
        print(f"Running Ablation {ablation_num}: {description}")
        print(f"{'='*80}")
        
        script_path = Path(__file__).parent / f"{script_name}.py"
        
        if not script_path.exists():
            print(f"❌ ERROR: Script not found: {script_path}")
            return False
        
        # Build command
        cmd = [
            'python',
            str(script_path),
            f'--output={str(self.output_dir / f"ablation_{ablation_num}.json")}',
        ]
        
        # Add ablation-specific arguments
        if ablation_num == 1:
            pass  # No special args needed
        elif ablation_num == 2:
            cmd.extend([
                '--arena-sizes', '64', '128', '256', '300',
                '--n-sessions', '1000',
                '--n-trials', '3',
            ])
        elif ablation_num == 3:
            cmd.extend([
                '--n-tokens', '100',
                '--n-trials', '3',
            ])
        elif ablation_num == 4:
            cmd.extend([
                '--n-trials', '3',
            ])
        
        t_start = time.time()
        
        try:
            # FIXED: Increased timeout - Ablations 2 & 4 need up to 10min per config
            # Ablation 2: 4 arena sizes × 2 modes × 10min = 80min
            # Ablation 4: 8 searches × 10min = 80min
            timeout_sec = 10800  # 3 hours per ablation (was 1 hour)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            
            t_end = time.time()
            duration = t_end - t_start
            self.timings[f'ablation_{ablation_num}'] = duration
            
            if result.returncode == 0:
                print(f"\n✅ Ablation {ablation_num} completed successfully ({duration:.1f}s)")
                
                # Try to load results
                output_file = self.output_dir / f"ablation_{ablation_num}.json"
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        self.results[f'ablation_{ablation_num}'] = json.load(f)
                
                return True
            else:
                print(f"\n❌ Ablation {ablation_num} FAILED")
                print(f"STDERR: {result.stderr[-1000:]}")
                return False
        
        except subprocess.TimeoutExpired:
            print(f"\n❌ Ablation {ablation_num} TIMEOUT (exceeded 3 hours)")
            return False
        except Exception as e:
            print(f"\n❌ Ablation {ablation_num} ERROR: {e}")
            return False
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of all ablations."""
        lines = [
            "\n" + "="*80,
            "OSDI ABLATION STUDY: SUMMARY REPORT",
            "="*80,
            f"\nResults directory: {self.output_dir}",
            f"\nCompletion time: {sum(self.timings.values()):.1f}s total",
            "",
            "Ablation Timings:",
        ]
        
        for ablation_name, duration in sorted(self.timings.items()):
            lines.append(f"  {ablation_name}: {duration:.1f}s")
        
        lines.extend([
            "",
            "Generated Files:",
            "  - LaTeX tables (.tex)",
            "  - Results JSON (.json)",
            "  - Figures (.pdf)",
        ])
        
        # List generated files
        gen_files = list(self.output_dir.glob("*.json")) + \
                    list(self.output_dir.glob("*.tex")) + \
                    list(self.output_dir.glob("*.pdf"))
        
        if gen_files:
            lines.append("\nGenerated artifacts:")
            for f in sorted(gen_files):
                lines.append(f"  - {f.name}")
        
        return "\n".join(lines)
    
    def run_all_ablations(self, skip_ablations: List[int] = None) -> bool:
        """
        Run all ablation studies in sequence.
        
        Args:
            skip_ablations: List of ablation numbers to skip
        
        Returns:
            True if all ablations completed successfully
        """
        skip_ablations = skip_ablations or []
        
        ablations = [
            (1, 'ablation_os_tax', 'OS Tax (Dispatch Overhead Analysis)'),
            (2, 'ablation_session_arena', 'Session Arena Decomposition'),
            (3, 'ablation_plan_cache', 'Plan Cache Effectiveness'),
            (4, 'ablation_semantic_signals', 'Semantic Signal Value'),
        ]
        
        results = {}
        
        for ablation_num, script_name, description in ablations:
            skip = ablation_num in skip_ablations
            success = self.run_ablation(ablation_num, script_name, description, skip=skip)
            results[ablation_num] = success
        
        return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description="Master runner for OSDI ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all ablations
  python run_all_ablations.py
  
  # Skip ablation 2 (takes longest)
  python run_all_ablations.py --skip-ablation 2
  
  # Custom output directory
  python run_all_ablations.py --output-dir /path/to/results
        """
    )
    
    parser.add_argument(
        '--skip-ablation',
        type=int,
        nargs='+',
        default=[],
        help='Ablation numbers to skip (1-4)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/ubuntu/Djinn/OSDI_Evaluation/ablation/results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("OSDI ABLATION STUDY: MASTER RUNNER")
    print("="*80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Skip ablations: {args.skip_ablation if args.skip_ablation else 'None'}")
    
    runner = AblationRunner(args.output_dir)
    
    # Run all ablations
    success = runner.run_all_ablations(skip_ablations=args.skip_ablation)
    
    # Print summary report
    print(runner.generate_summary_report())
    
    # Final status
    print("\n" + "="*80)
    if success:
        print("✅ ALL ABLATIONS COMPLETED SUCCESSFULLY")
    else:
        print("⚠️  SOME ABLATIONS FAILED - SEE ABOVE FOR DETAILS")
    print("="*80)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
