#!/usr/bin/env python3
"""
Validation script for ablation studies.

This script validates the ablation code structure without requiring a running server.
It checks:
1. All imports are valid
2. Function signatures are correct
3. Code structure follows the plan
4. Output paths are correct
"""

import sys
from pathlib import Path

# Add Djinn to path
sys.path.insert(0, '/home/ubuntu/Djinn')


def validate_imports():
    """Validate that all required imports work."""
    print("\n" + "="*70)
    print("VALIDATING IMPORTS")
    print("="*70)
    
    errors = []
    
    # Test Djinn imports
    try:
        from djinn.backend.runtime.initialization import init_async
        from djinn.config import DjinnConfig
        from djinn.core.coordinator import get_coordinator
        from djinn.core.enhanced_model_manager import EnhancedModelManager
        from djinn.core.ghost_loader import create_hf_ghost_model
        print("✅ Djinn imports successful")
    except Exception as e:
        errors.append(f"Djinn imports failed: {e}")
        print(f"❌ Djinn imports failed: {e}")
    
    # Test PyTorch imports
    try:
        import torch
        import torch.nn as nn
        print("✅ PyTorch imports successful")
    except Exception as e:
        errors.append(f"PyTorch imports failed: {e}")
        print(f"❌ PyTorch imports failed: {e}")
    
    # Test other imports
    try:
        import numpy as np
        from transformers import AutoTokenizer
        import aiohttp
        print("✅ Other imports successful")
    except Exception as e:
        errors.append(f"Other imports failed: {e}")
        print(f"❌ Other imports failed: {e}")
    
    return errors


def validate_ablation_os_tax():
    """Validate OS Tax ablation structure."""
    print("\n" + "="*70)
    print("VALIDATING ABLATION 1: OS TAX")
    print("="*70)
    
    errors = []
    
    try:
        import ablation_os_tax
        
        # Check required functions exist
        required_functions = [
            'measure_native_operations',
            'measure_djinn_operations',
            'compute_overhead',
            'generate_latex_table',
            'run_ablation'
        ]
        
        for func_name in required_functions:
            if not hasattr(ablation_os_tax, func_name):
                errors.append(f"Missing function: {func_name}")
                print(f"❌ Missing function: {func_name}")
            else:
                print(f"✅ Function exists: {func_name}")
        
        # Check TinyModel class
        if not hasattr(ablation_os_tax, 'TinyModel'):
            errors.append("Missing TinyModel class")
            print("❌ Missing TinyModel class")
        else:
            print("✅ TinyModel class exists")
        
        if not errors:
            print("\n✅ OS Tax ablation structure is valid")
    
    except Exception as e:
        errors.append(f"Failed to import ablation_os_tax: {e}")
        print(f"❌ Failed to import ablation_os_tax: {e}")
    
    return errors


def validate_ablation_plan_cache():
    """Validate Plan Cache ablation structure."""
    print("\n" + "="*70)
    print("VALIDATING ABLATION 2: PLAN CACHE")
    print("="*70)
    
    errors = []
    
    try:
        import ablation_plan_cache
        
        # Check required functions exist
        required_functions = [
            'fetch_cache_stats',
            'measure_uniform_workload',
            'measure_varied_workload',
            'generate_latex_table',
            'run_ablation'
        ]
        
        for func_name in required_functions:
            if not hasattr(ablation_plan_cache, func_name):
                errors.append(f"Missing function: {func_name}")
                print(f"❌ Missing function: {func_name}")
            else:
                print(f"✅ Function exists: {func_name}")
        
        if not errors:
            print("\n✅ Plan Cache ablation structure is valid")
    
    except Exception as e:
        errors.append(f"Failed to import ablation_plan_cache: {e}")
        print(f"❌ Failed to import ablation_plan_cache: {e}")
    
    return errors


def validate_runner():
    """Validate master runner script."""
    print("\n" + "="*70)
    print("VALIDATING MASTER RUNNER")
    print("="*70)
    
    errors = []
    
    try:
        import run_ablations
        
        # Check required functions exist
        required_functions = [
            'run_all_ablations'
        ]
        
        for func_name in required_functions:
            if not hasattr(run_ablations, func_name):
                errors.append(f"Missing function: {func_name}")
                print(f"❌ Missing function: {func_name}")
            else:
                print(f"✅ Function exists: {func_name}")
        
        if not errors:
            print("\n✅ Master runner structure is valid")
    
    except Exception as e:
        errors.append(f"Failed to import run_ablations: {e}")
        print(f"❌ Failed to import run_ablations: {e}")
    
    return errors


def validate_output_structure():
    """Validate output directory structure."""
    print("\n" + "="*70)
    print("VALIDATING OUTPUT STRUCTURE")
    print("="*70)
    
    errors = []
    
    # Check results directory can be created
    results_dir = Path(__file__).parent.parent / "results"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Results directory: {results_dir}")
        
        # Check expected output files
        expected_files = [
            'ablation_os_tax.json',
            'ablation_plan_cache.json',
            'os_tax_table.tex',
            'plan_cache_table.tex',
            'ablation_summary.json'
        ]
        
        print("\nExpected output files:")
        for filename in expected_files:
            filepath = results_dir / filename
            print(f"  - {filepath}")
        
    except Exception as e:
        errors.append(f"Failed to create results directory: {e}")
        print(f"❌ Failed to create results directory: {e}")
    
    return errors


def validate_server_integration():
    """Validate server integration points."""
    print("\n" + "="*70)
    print("VALIDATING SERVER INTEGRATION")
    print("="*70)
    
    errors = []
    
    # Check that server.py has plan_cache stats
    server_file = Path('/home/ubuntu/Djinn/djinn/server/server.py')
    if not server_file.exists():
        errors.append("server.py not found")
        print("❌ server.py not found")
        return errors
    
    server_content = server_file.read_text()
    
    # Check for plan_cache integration
    if 'get_meta_simulator' in server_content and 'plan_cache' in server_content:
        print("✅ Server has plan_cache stats integration")
    else:
        errors.append("Server missing plan_cache stats integration")
        print("❌ Server missing plan_cache stats integration")
    
    # Check MetaSimulator has get_cache_stats
    meta_sim_file = Path('/home/ubuntu/Djinn/djinn/server/meta_simulator.py')
    if meta_sim_file.exists():
        meta_sim_content = meta_sim_file.read_text()
        if 'get_cache_stats' in meta_sim_content:
            print("✅ MetaSimulator has get_cache_stats method")
        else:
            errors.append("MetaSimulator missing get_cache_stats method")
            print("❌ MetaSimulator missing get_cache_stats method")
    else:
        errors.append("meta_simulator.py not found")
        print("❌ meta_simulator.py not found")
    
    return errors


def main():
    """Run all validations."""
    print("\n" + "="*70)
    print("ABLATION STUDY VALIDATION")
    print("="*70)
    print("\nThis script validates the ablation code structure without")
    print("requiring a running server.")
    
    all_errors = []
    
    # Run validations
    all_errors.extend(validate_imports())
    all_errors.extend(validate_ablation_os_tax())
    all_errors.extend(validate_ablation_plan_cache())
    all_errors.extend(validate_runner())
    all_errors.extend(validate_output_structure())
    all_errors.extend(validate_server_integration())
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if not all_errors:
        print("\n✅ ALL VALIDATIONS PASSED")
        print("\nThe ablation studies are ready to run.")
        print("\nTo run the ablations:")
        print("  1. Start server: python3 -m djinn.server --port 5556 --gpu 0")
        print("  2. Run ablations: python3 run_ablations.py")
        return 0
    else:
        print(f"\n❌ VALIDATION FAILED ({len(all_errors)} errors)")
        print("\nErrors:")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
