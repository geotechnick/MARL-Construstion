#!/usr/bin/env python3
"""
Installation script for MARL Construction Simulation.
This script sets up the environment and installs all dependencies.
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"⚙️  {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        return False
    else:
        print(f"✅ Python version {python_version.major}.{python_version.minor}.{python_version.micro} is compatible!")
        return True

def check_gpu_availability():
    """Check if CUDA is available for GPU acceleration."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU acceleration available!")
            print(f"   GPU Count: {gpu_count}")
            print(f"   GPU Name: {gpu_name}")
            return True
        else:
            print("💻 No GPU detected. Training will use CPU (slower but functional).")
            return False
    except ImportError:
        print("⚠️  PyTorch not yet installed. GPU check will be performed after installation.")
        return None

def install_system_dependencies():
    """Install system-level dependencies based on OS."""
    system = platform.system().lower()
    
    if system == "linux":
        print("🐧 Detected Linux system")
        # Check for pygame dependencies
        commands = [
            ("sudo apt-get update", "Updating package list"),
            ("sudo apt-get install -y python3-dev python3-pip", "Installing Python development tools"),
            ("sudo apt-get install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev", "Installing SDL2 for pygame")
        ]
        
        for command, description in commands:
            if not run_command(command, description):
                print("⚠️  Some system dependencies might not be installed. Continuing anyway...")
                
    elif system == "darwin":  # macOS
        print("🍎 Detected macOS system")
        # Check if Homebrew is installed
        if run_command("which brew", "Checking for Homebrew"):
            run_command("brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf", "Installing SDL2 via Homebrew")
        else:
            print("⚠️  Homebrew not found. Please install SDL2 manually if you encounter issues.")
            
    elif system == "windows":
        print("🪟 Detected Windows system")
        print("   No additional system dependencies required for Windows.")
    else:
        print(f"❓ Unknown system: {system}. Proceeding with Python package installation only.")

def create_virtual_environment():
    """Create and activate a virtual environment."""
    venv_name = "marl_env"
    
    if os.path.exists(venv_name):
        print(f"📁 Virtual environment '{venv_name}' already exists.")
        return venv_name
    
    if run_command(f"python -m venv {venv_name}", f"Creating virtual environment '{venv_name}'"):
        print(f"✅ Virtual environment created: {venv_name}")
        print(f"   To activate: ")
        
        if platform.system().lower() == "windows":
            print(f"   {venv_name}\\Scripts\\activate")
        else:
            print(f"   source {venv_name}/bin/activate")
        
        return venv_name
    else:
        print("❌ Failed to create virtual environment. Continuing with system Python.")
        return None

def install_python_packages():
    """Install Python packages from requirements.txt."""
    if os.path.exists("requirements.txt"):
        return run_command("pip install -r requirements.txt", "Installing Python packages from requirements.txt")
    else:
        print("📦 requirements.txt not found. Installing packages manually...")
        packages = [
            "torch>=2.0.0",
            "numpy>=1.21.0", 
            "pettingzoo>=1.24.0",
            "gymnasium>=0.29.0",
            "matplotlib>=3.5.0",
            "pygame>=2.1.0",
            "tensorboard>=2.10.0",
            "scipy>=1.9.0",
            "seaborn>=0.11.0",
            "tqdm>=4.64.0"
        ]
        
        for package in packages:
            if not run_command(f"pip install {package}", f"Installing {package}"):
                print(f"⚠️  Failed to install {package}. You may need to install it manually.")
        
        return True

def install_package_in_development_mode():
    """Install the current package in development mode."""
    return run_command("pip install -e .", "Installing MARL Construction package in development mode")

def verify_installation():
    """Verify that the installation was successful."""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        import numpy
        print("✅ NumPy imported successfully")
        
        import torch
        print("✅ PyTorch imported successfully")
        
        import pygame
        print("✅ Pygame imported successfully")
        
        import pettingzoo
        print("✅ PettingZoo imported successfully")
        
        # Test our package
        from marl_construction.envs import ConstructionEnv
        from marl_construction.agents import ConstructionAgent
        print("✅ MARL Construction package imported successfully")
        
        # Quick environment test
        env = ConstructionEnv(num_agents=2, site_width=5, site_height=5)
        obs, _ = env.reset()
        print("✅ Environment creation and reset successful")
        
        print("\n🎉 Installation verification completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error during verification: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

def main():
    """Main installation process."""
    print("=" * 60)
    print("🏗️  MARL Construction Simulation - Installation Script")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Install system dependencies (optional)
    print("\n📋 Installing system dependencies...")
    install_system_dependencies()
    
    # Step 3: Create virtual environment (optional but recommended)
    print("\n🐍 Setting up Python environment...")
    venv_name = create_virtual_environment()
    
    if venv_name:
        print(f"\n⚠️  Please activate the virtual environment before continuing:")
        if platform.system().lower() == "windows":
            print(f"   {venv_name}\\Scripts\\activate")
        else:
            print(f"   source {venv_name}/bin/activate")
        print("   Then run this script again, or continue with manual installation.")
        
        choice = input("\n❓ Continue with installation in current environment? (y/N): ").strip().lower()
        if choice not in ['y', 'yes']:
            print("Installation paused. Please activate the virtual environment and run again.")
            sys.exit(0)
    
    # Step 4: Install Python packages
    print("\n📦 Installing Python packages...")
    if not install_python_packages():
        print("❌ Failed to install some packages. Please check the errors above.")
        sys.exit(1)
    
    # Step 5: Install current package
    print("\n🔧 Installing MARL Construction package...")
    if not install_package_in_development_mode():
        print("❌ Failed to install the package. Please check the errors above.")
        sys.exit(1)
    
    # Step 6: Verify installation
    print("\n✅ Verifying installation...")
    if verify_installation():
        print("\n🎉 Installation completed successfully!")
        
        # Check GPU
        check_gpu_availability()
        
        print("\n🚀 Next steps:")
        print("   1. Run a basic training example:")
        print("      python examples/basic_training.py")
        print("   2. Check out the documentation in README.md")
        print("   3. Explore the examples/ directory for more scripts")
        
    else:
        print("\n❌ Installation verification failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()