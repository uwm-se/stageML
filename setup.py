from setuptools import setup, find_packages

setup(
    name="stageml",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24",
    ],
    description="Multi-Stage Programming DSL for ML Inference with MLIR backend",
    author="Xavier",
)
