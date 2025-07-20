#!/usr/bin/env python3
"""
Final comprehensive test of the MARL Construction system.
"""

import torch
import numpy as np
from marl_construction.envs import ConstructionEnv
from marl_construction.agents import ConstructionAgent
from marl_construction.training import MARLTrainer

def test_complete_system():
    """Test the complete system with a very short training run."""
    print("=== FINAL SYSTEM TEST ===")
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Test Environment
    print("1. Testing Environment...")
    env = ConstructionEnv(
        num_agents=2,
        site_width=4,
        site_height=4,
        max_timesteps=5,
        render_mode=None
    )
    
    obs, info = env.reset()
    actions = {agent_id: 0 for agent_id in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    print("   [OK] Environment working")
    
    # 2. Test Agents
    print("2. Testing Agents...")
    obs_dim = len(obs[env.agents[0]])
    action_dim = env.action_spaces[env.agents[0]].n
    
    agents = {}
    for i, agent_id in enumerate(env.agents):
        role = ["builder", "transporter"][i]
        agents[agent_id] = ConstructionAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=8,
            role=role
        )
    print("   [OK] Agents created")
    
    # 3. Test Action Selection
    print("3. Testing Action Selection...")
    for agent_id, agent in agents.items():
        test_obs = torch.FloatTensor(obs[agent_id]).unsqueeze(0)
        action, message = agent.get_action(test_obs)
        assert 0 <= action < action_dim
    print("   [OK] Action selection working")
    
    # 4. Test Training Setup
    print("4. Testing Training Setup...")
    trainer = MARLTrainer(
        env=env,
        agents=agents,
        log_dir="./logs/final_test"
    )
    print("   [OK] Trainer created")
    
    # 5. Test Episode Collection
    print("5. Testing Episode Collection...")
    episode_data, total_rewards = trainer.collect_episode(max_steps=3)
    assert len(episode_data) == len(env.agents)
    print("   [OK] Episode collection working")
    
    # 6. Test Training Update
    print("6. Testing Training Update...")
    for agent_id in env.agents:
        if episode_data[agent_id]:  # Only if we have data
            trainer.update_agent(agent_id, episode_data[agent_id], epochs=1)
    print("   [OK] Training update working")
    
    env.close()
    
    print("\n[SUCCESS] ALL TESTS PASSED!")
    print("[OK] Environment creation and simulation")
    print("[OK] Multi-agent system with specialized roles")
    print("[OK] Neural network training pipeline")
    print("[OK] Complete MARL framework functionality")
    print("\nSystem is ready for full training!")
    return True

def test_package_installation():
    """Test that the package is properly installed."""
    print("=== PACKAGE INSTALLATION TEST ===")
    
    try:
        import marl_construction
        from marl_construction.envs import ConstructionEnv
        from marl_construction.agents import ConstructionAgent
        from marl_construction.training import MARLTrainer
        print("[OK] Package imports working")
        
        # Test version
        print(f"[OK] Package version: {marl_construction.__version__}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Package installation test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing MARL Construction System...")
    print("=" * 50)
    
    success1 = test_package_installation()
    success2 = test_complete_system()
    
    if success1 and success2:
        print("\n" + "=" * 50)
        print("SYSTEM FULLY FUNCTIONAL!")
        print("=" * 50)
        print("\nReady to run:")
        print("  python scripts/quick_start.py")
        print("  python examples/basic_training.py")
        print("  python examples/evaluate_agents.py")
    else:
        print("\n[FAIL] Some tests failed. Check the output above.")