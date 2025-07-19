from setuptools import setup, find_packages

setup(
    name="marl-construction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "pettingzoo>=1.24.0",
        "ray[rllib]>=2.8.0",
        "gymnasium>=0.29.0",
        "matplotlib>=3.5.0",
        "pygame>=2.1.0",
        "tensorboard>=2.10.0",
        "scipy>=1.9.0",
        "networkx>=2.8.0",
        "seaborn>=0.11.0",
        "tqdm>=4.64.0",
    ],
    python_requires=">=3.8",
    author="MARL Construction Team",
    description="Multi-Agent Reinforcement Learning for Construction Simulation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)