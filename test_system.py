#!/usr/bin/env python3
"""
Test script to verify the MARL Construction system works correctly.
This script runs comprehensive tests on all components.
"""

import sys
import traceback
import numpy as np
import torch

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test standard dependencies
        import numpy as np
        print("  [OK] NumPy imported")
        
        import torch
        print("  [OK] PyTorch imported")
        
        import pygame
        print("  [OK] Pygame imported")
        
        import matplotlib.pyplot as plt
        print("  [OK] Matplotlib imported")
        
        # Test our package imports
        from marl_construction.envs import ConstructionEnv
        print("  [OK] ConstructionEnv imported")
        
        from marl_construction.agents import ConstructionAgent
        print("  [OK] ConstructionAgent imported")
        
        from marl_construction.training import MARLTrainer
        print("  [OK] MARLTrainer imported")
        
        print("[PASS] All imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during imports: {e}")
        return False

def test_environment():
    """Test environment creation and basic functionality."""
    print("Testing environment...")
    
    try:
        from marl_construction.envs import ConstructionEnv
        
        # Create environment
        env = ConstructionEnv(
            num_agents=2,
            site_width=5,
            site_height=5,
            max_timesteps=10,
            render_mode=None  # No rendering for tests
        )
        print("  [OK] Environment created")
        
        # Test reset
        observations, infos = env.reset()
        print("  [OK] Environment reset successful")
        
        # Check observation format
        assert len(observations) == env.num_agents, "Wrong number of agents"
        for agent_id, obs in observations.items():
            assert isinstance(obs, np.ndarray), "Observation not numpy array"
            assert len(obs) > 0, "Empty observation"
        print("  [OK] Observations format correct")
        
        # Test step
        actions = {agent_id: 0 for agent_id in env.agents}  # All stay still
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print("  [OK] Environment step successful")
        
        # Check step outputs
        assert len(rewards) == env.num_agents, "Wrong number of rewards"
        assert len(terminations) == env.num_agents, "Wrong number of terminations"
        print("  [OK] Step outputs format correct")
        
        env.close()
        print("[PASS] Environment test passed!\n")
        return True
        
    except Exception as e:
        print(f"[FAIL] Environment test failed: {e}")
        traceback.print_exc()
        return False

def test_agents():
    """Test agent creation and action selection."""
    print("Testing agents...")
    
    try:
        from marl_construction.agents import ConstructionAgent
        
        # Test agent creation
        obs_dim = 50
        action_dim = 10
        
        roles = ["builder", "transporter", "crane_operator", "supervisor"]
        agents = {}
        
        for i, role in enumerate(roles):
            agent = ConstructionAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=32,  # Small for testing
                role=role
            )
            agents[f"agent_{i}"] = agent
            print(f"  ✅ {role} agent created")
        
        # Test forward pass
        test_obs = torch.randn(1, obs_dim)
        for agent_id, agent in agents.items():
            action_probs, value, message = agent(test_obs)
            
            # Check outputs
            assert action_probs.shape == (1, action_dim), "Wrong action_probs shape"
            assert value.shape == (1, 1), "Wrong value shape"
            assert message.shape == (1, 16), "Wrong message shape"
            print(f"  ✅ {agent_id} forward pass successful")
        
        # Test action selection
        for agent_id, agent in agents.items():
            action, message = agent.get_action(test_obs, deterministic=True)
            assert isinstance(action, int), "Action not integer"
            assert 0 <= action < action_dim, "Action out of bounds"
            print(f"  ✅ {agent_id} action selection successful")
        
        print("✅ Agent test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        traceback.print_exc()
        return False

def test_training_setup():
    """Test training pipeline setup."""
    print("🏋️ Testing training setup...")
    
    try:
        from marl_construction.envs import ConstructionEnv
        from marl_construction.agents import ConstructionAgent
        from marl_construction.training import MARLTrainer
        
        # Create small environment for testing
        env = ConstructionEnv(
            num_agents=2,
            site_width=3,
            site_height=3,
            max_timesteps=5,
            render_mode=None
        )
        
        # Get dimensions
        sample_obs, _ = env.reset()
        obs_dim = len(sample_obs[env.agents[0]])
        action_dim = env.action_spaces[env.agents[0]].n
        
        # Create agents
        agents = {}
        for i, agent_id in enumerate(env.agents):
            role = ["builder", "transporter"][i]
            agents[agent_id] = ConstructionAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=16,  # Very small for testing
                role=role
            )
        print("  ✅ Test agents created")
        
        # Create trainer
        trainer = MARLTrainer(
            env=env,
            agents=agents,
            lr=1e-3,
            log_dir="./logs/test"
        )
        print("  ✅ Trainer created")
        
        # Test episode collection (short episode)
        episode_data, total_rewards = trainer.collect_episode(max_steps=3)
        print("  ✅ Episode collection successful")
        
        # Check episode data format
        for agent_id in env.agents:
            assert agent_id in episode_data, f"Missing data for {agent_id}"
            assert len(episode_data[agent_id]) > 0, f"Empty data for {agent_id}"
            
            # Check first step data
            step_data = episode_data[agent_id][0]
            required_keys = ['observation', 'action', 'value', 'log_prob', 'reward']
            for key in required_keys:
                assert key in step_data, f"Missing {key} in step data"
        print("  ✅ Episode data format correct")
        
        env.close()
        print("✅ Training setup test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Training setup test failed: {e}")
        traceback.print_exc()
        return False

def test_quick_training():
    """Test a very short training run."""
    print("⚡ Testing quick training...")
    
    try:
        from marl_construction.envs import ConstructionEnv
        from marl_construction.agents import ConstructionAgent
        from marl_construction.training import MARLTrainer
        
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create minimal environment
        env = ConstructionEnv(
            num_agents=2,
            site_width=3,
            site_height=3,
            max_timesteps=5,
            render_mode=None
        )
        
        # Get dimensions
        sample_obs, _ = env.reset()
        obs_dim = len(sample_obs[env.agents[0]])
        action_dim = env.action_spaces[env.agents[0]].n
        
        # Create agents
        agents = {}
        for i, agent_id in enumerate(env.agents):
            role = ["builder", "transporter"][i]
            agents[agent_id] = ConstructionAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=8,  # Tiny for speed
                role=role
            )
        
        # Create trainer
        trainer = MARLTrainer(
            env=env,
            agents=agents,
            lr=1e-2,  # Higher LR for quick test
            log_dir="./logs/test_quick"
        )
        
        # Run very short training
        print("  🏃 Running 3 episodes of training...")
        episode_rewards = trainer.train(
            num_episodes=3,
            update_frequency=2,
            save_frequency=10
        )
        print("  ✅ Quick training completed")
        
        # Check results
        assert len(episode_rewards) == len(env.agents), "Missing agent rewards"
        for agent_id, rewards in episode_rewards.items():
            assert len(rewards) > 0, f"No rewards for {agent_id}"
        print("  ✅ Training results format correct")
        
        env.close()
        print("✅ Quick training test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("MARL Construction System - Comprehensive Testing")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Environment", test_environment),
        ("Agents", test_agents),
        ("Training Setup", test_training_setup),
        ("Quick Training", test_quick_training)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED! The system is working correctly.")
        print("\n🚀 Ready to run:")
        print("   python scripts/quick_start.py")
        print("   python examples/basic_training.py")
        return True
    else:
        print(f"\n⚠️  {len(tests) - passed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)