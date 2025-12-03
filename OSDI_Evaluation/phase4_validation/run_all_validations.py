#!/usr/bin/env python3
"""
Phase 4 Validation: Unified Test Runner

Runs all Phase 4 validation tests in sequence:
1. Logit Equivalence (Ring Buffer correctness)
2. PCIe Flatline (Bandwidth saturation)
3. Agent Sleep/Resume (Semantic Scheduler lifecycle)

Usage:
    python run_all_validations.py --config configs/validation_smoke.yaml
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ValidationRunner:
    """Unified validation test runner."""
    
    def __init__(self, config_file: str):
        """Initialize with config file."""
        self.config_file = Path(config_file)
        self.results = {}
        self.summary = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
        }
        
        # Load config
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"✅ Loaded config: {self.config_file}")
        logger.info(f"   Experiment: {self.config['experiment']['name']}")
        logger.info(f"   Description: {self.config['experiment']['description']}")
    
    async def run_logit_equivalence(self) -> bool:
        """Run logit equivalence test."""
        test_config = self.config.get('logit_equivalence', {})
        
        if not test_config.get('enabled', False):
            logger.info("\n⏭️  Skipping logit equivalence test")
            self.summary["skipped"] += 1
            return None
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST 1: LOGIT EQUIVALENCE")
        logger.info("=" * 80)
        
        self.summary["total"] += 1
        
        try:
            from OSDI_Evaluation.phase4_validation.scripts.test_logit_equivalence import main as test_main
            
            # Save original argv
            original_argv = sys.argv.copy()
            
            # Set argv for test
            sys.argv = [
                'test_logit_equivalence.py',
                '--model', test_config.get('model_id', 'meta-llama/Llama-2-7b-hf'),
                '--num-samples', str(test_config.get('num_samples', 5)),
                '--output-dir', self.config['logging'].get('results_dir', 'OSDI_Evaluation/phase4_validation/results'),
            ]
            
            # Run test
            result = await test_main()
            
            # Restore argv
            sys.argv = original_argv
            
            if result:
                logger.info("✅ Test 1 PASSED")
                self.summary["passed"] += 1
                self.results["logit_equivalence"] = "PASSED"
            else:
                logger.error("❌ Test 1 FAILED")
                self.summary["failed"] += 1
                self.results["logit_equivalence"] = "FAILED"
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Test 1 ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.summary["failed"] += 1
            self.summary["errors"].append(f"Logit equivalence: {str(e)}")
            self.results["logit_equivalence"] = "ERROR"
            return False
    
    async def run_pcie_flatline(self) -> bool:
        """Run PCIe flatline test."""
        test_config = self.config.get('pcie_flatline', {})
        
        if not test_config.get('enabled', False):
            logger.info("\n⏭️  Skipping PCIe flatline test")
            self.summary["skipped"] += 1
            return None
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST 2: PCIe FLATLINE")
        logger.info("=" * 80)
        
        self.summary["total"] += 1
        
        try:
            from OSDI_Evaluation.phase4_validation.scripts.test_pcie_flatline import main as test_main
            
            # Save original argv
            original_argv = sys.argv.copy()
            
            # Set argv for test
            sys.argv = [
                'test_pcie_flatline.py',
                '--model', test_config.get('model_id', 'meta-llama/Llama-2-70b-hf'),
                '--duration', str(test_config.get('sample_duration_seconds', 10)),
                '--min-bw', str(test_config.get('min_sustained_bandwidth_mb_per_s', 20000)),
                '--output-dir', self.config['logging'].get('results_dir', 'OSDI_Evaluation/phase4_validation/results'),
            ]
            
            # Run test
            result = await test_main()
            
            # Restore argv
            sys.argv = original_argv
            
            if result:
                logger.info("✅ Test 2 PASSED")
                self.summary["passed"] += 1
                self.results["pcie_flatline"] = "PASSED"
            else:
                logger.error("❌ Test 2 FAILED")
                self.summary["failed"] += 1
                self.results["pcie_flatline"] = "FAILED"
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Test 2 ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.summary["failed"] += 1
            self.summary["errors"].append(f"PCIe flatline: {str(e)}")
            self.results["pcie_flatline"] = "ERROR"
            return False
    
    async def run_agent_sleep_resume(self) -> bool:
        """Run agent sleep/resume test."""
        test_config = self.config.get('agent_sleep_resume', {})
        
        if not test_config.get('enabled', False):
            logger.info("\n⏭️  Skipping agent sleep/resume test")
            self.summary["skipped"] += 1
            return None
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST 3: AGENT SLEEP/RESUME")
        logger.info("=" * 80)
        
        self.summary["total"] += 1
        
        try:
            from OSDI_Evaluation.phase4_validation.scripts.test_agent_sleep_resume import main as test_main
            
            # Save original argv
            original_argv = sys.argv.copy()
            
            # Set argv for test
            sys.argv = [
                'test_agent_sleep_resume.py',
                '--model', test_config.get('model_id', 'meta-llama/Llama-2-7b-hf'),
                '--sleep', str(test_config.get('sleep_duration_seconds', 30)),
                '--output-dir', self.config['logging'].get('results_dir', 'OSDI_Evaluation/phase4_validation/results'),
            ]
            
            # Run test
            result = await test_main()
            
            # Restore argv
            sys.argv = original_argv
            
            if result:
                logger.info("✅ Test 3 PASSED")
                self.summary["passed"] += 1
                self.results["agent_sleep_resume"] = "PASSED"
            else:
                logger.error("❌ Test 3 FAILED")
                self.summary["failed"] += 1
                self.results["agent_sleep_resume"] = "FAILED"
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Test 3 ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.summary["failed"] += 1
            self.summary["errors"].append(f"Agent sleep/resume: {str(e)}")
            self.results["agent_sleep_resume"] = "ERROR"
            return False
    
    async def run_all(self) -> bool:
        """Run all validations."""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: METRIC FIDELITY & CORRECTNESS VALIDATION")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Run tests
        await self.run_logit_equivalence()
        await self.run_pcie_flatline()
        await self.run_agent_sleep_resume()
        
        elapsed = time.time() - start_time
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\nResults:")
        logger.info(f"  Total tests: {self.summary['total']}")
        logger.info(f"  Passed: {self.summary['passed']}")
        logger.info(f"  Failed: {self.summary['failed']}")
        logger.info(f"  Skipped: {self.summary['skipped']}")
        
        if self.summary['errors']:
            logger.info(f"\nErrors:")
            for error in self.summary['errors']:
                logger.info(f"  - {error}")
        
        logger.info(f"\nDuration: {elapsed:.1f}s")
        
        # Overall result
        overall_passed = self.summary['failed'] == 0 and self.summary['total'] > 0
        
        if overall_passed:
            logger.info("\n" + "🎉 " * 20)
            logger.info("✅ ALL VALIDATIONS PASSED")
            logger.info("🎉 " * 20)
        else:
            logger.error("\n" + "❌ " * 20)
            logger.error("SOME VALIDATIONS FAILED")
            logger.error("❌ " * 20)
        
        # Save summary
        results_dir = Path(self.config['logging'].get('results_dir', 'OSDI_Evaluation/phase4_validation/results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = results_dir / "validation_summary.json"
        summary_data = {
            "config": str(self.config_file),
            "experiment": self.config['experiment'],
            "timestamp": time.time(),
            "duration_seconds": elapsed,
            "summary": self.summary,
            "results": self.results,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"\n📄 Summary saved to: {summary_file}")
        
        return overall_passed


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Phase 4: Unified Validation Runner")
    parser.add_argument('--config', type=str, required=True,
                       help='Configuration file (YAML)')
    
    args = parser.parse_args()
    
    try:
        runner = ValidationRunner(args.config)
        passed = await runner.run_all()
        sys.exit(0 if passed else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

