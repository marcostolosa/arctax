#!/usr/bin/env python3
"""
Setup script para instalação do Arctax CLI
Permite usar: $ arctax <param1> <param2> <param3>
"""

from setuptools import setup, find_packages

setup(
    name="arctax",
    version="1.0.0",
    description="Sistema avançado de geração de bypass prompts baseado na Arcanum Taxonomy",
    author="Arctax Team",
    packages=find_packages(),
    install_requires=[
        "typer[all]>=0.9.0",
        "rich>=13.0.0", 
        "httpx>=0.25.0",
        "pydantic>=2.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "joblib>=1.3.0",
        "markdown>=3.5.0",
        "beautifulsoup4>=4.12.0",
        "pyyaml>=6.0.0"
    ],
    entry_points={
        "console_scripts": [
            "arctax=arctax.cli:app",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Security Researchers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)