#!/usr/bin/env python3
"""Setup script for Djinn - GPU Disaggregation Framework for PyTorch."""

from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
def read_readme():
    """Read README.md for long description."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="djinn-gpu",
    version="0.1.0",
    description="GPU Disaggregation Framework for PyTorch (Python-only, Zero C++ Dependencies)",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Djinn Team",
    author_email="",
    url="https://github.com/yourusername/djinn",
    license="MIT",

    # Package discovery
    packages=find_packages(exclude=['tests', 'tests.*', 'benchmarks', 'example', 'docs', 'scripts']),

    # Include optional pre-built extensions (if available)
    package_data={
        'djinn': [
            '*.so',
            '_C*.so',
            '_runtime*.so',
        ],
    },
    
    # Dependencies
    install_requires=read_requirements('requirements.txt'),
    
    extras_require={
        'dev': read_requirements('requirements-dev.txt') if os.path.exists('requirements-dev.txt') else [],
        'evaluation': [
            'pyyaml>=6.0.0',
            'pynvml>=11.0.0',
        ],
    },
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Distributed Computing',
    ],
    
    # Entry points (optional - for command-line tools)
    entry_points={
        'console_scripts': [
            # Add command-line tools here if needed
            # 'djinn-server=djinn.backend.server.server:main',
        ],
    },
)

