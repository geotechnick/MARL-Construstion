# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### Site Builder Issues

#### Problem: `pygame.error: display Surface quit`
**Cause**: Display initialization issues on some systems

**Solutions**:
1. **Use Text Builder**: `python examples/text_site_builder.py`
2. **Check Display**: Ensure you have a display available (not running headless)
3. **Update Pygame**: `pip install --upgrade pygame`

#### Problem: "No display" errors
**Solutions**:
- **Windows**: Make sure you're not running in WSL without X11
- **Linux**: Install display packages: `sudo apt-get install python3-tk`
- **Remote**: Use text builder for headless systems

### Training Issues

#### Problem: "CUDA out of memory"
**Solutions**:
1. Reduce batch size: Use smaller `update_frequency` in training
2. Reduce network size: Use smaller `hidden_dim` in agents
3. Use CPU: Training works fine on CPU, just slower

#### Problem: Slow training
**Solutions**:
1. **Reduce environment size**: Use smaller `site_width` and `site_height`
2. **Fewer agents**: Start with 2-3 agents instead of 4
3. **Shorter episodes**: Reduce `max_timesteps`

### Installation Issues

#### Problem: Import errors
**Solutions**:
1. **Install in development mode**: `pip install -e .`
2. **Check dependencies**: `pip install -r requirements.txt`
3. **Python version**: Ensure Python 3.8+

#### Problem: Ray/RLlib issues
**Solutions**:
1. **Skip Ray**: The system works without Ray for basic training
2. **Downgrade**: `pip install ray==2.8.0` if newer versions have issues
3. **CPU only**: Ray works fine on CPU

## System Requirements

### Minimum
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 1GB free space
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04+

### Recommended
- **Python**: 3.9+
- **RAM**: 8GB+
- **GPU**: CUDA-compatible (optional)
- **Display**: For GUI site builder

## Alternative Workflows

### Headless/Server Systems
```bash
# Use text-based builder
python examples/text_site_builder.py

# Train without rendering
python examples/basic_training.py  # Will work without display

# Evaluate without rendering (set render_mode=None in code)
```

### Low-Memory Systems
```bash
# Small environment
python examples/basic_training.py --agents 2 --width 8 --height 8

# Quick training
python scripts/quick_start.py --mode train
```

### Docker/Container Systems
```dockerfile
# Add to Dockerfile for display support
RUN apt-get update && apt-get install -y \
    python3-tk \
    xvfb

# Run with virtual display
ENV DISPLAY=:99
RUN Xvfb :99 -screen 0 1024x768x24 &
```

## Performance Tips

### Faster Training
1. **Start small**: 2 agents, 8x8 grid, 50 episodes
2. **Profile code**: Use `python -m cProfile script.py`
3. **Batch operations**: Use higher `update_frequency`

### Memory Optimization
1. **Clear logs**: Delete old tensorboard logs
2. **Limit history**: Reduce episode buffer sizes
3. **Close environments**: Always call `env.close()`

## Getting Help

### Debug Information
When reporting issues, include:
```bash
# System info
python --version
pip list | grep -E "(torch|pygame|pettingzoo|ray)"

# Test basic functionality
python test_final.py
```

### Common Commands for Debugging
```bash
# Check imports
python -c "from marl_construction.envs import ConstructionEnv; print('OK')"

# Test display
python -c "import pygame; pygame.init(); print('Pygame OK')"

# Test environment
python -c "
from marl_construction.envs import ConstructionEnv
env = ConstructionEnv(num_agents=2, site_width=5, site_height=5, render_mode=None)
obs, _ = env.reset()
print('Environment OK')
env.close()
"
```

## Error Messages Reference

| Error | Meaning | Solution |
|-------|---------|----------|
| `pygame.error: display Surface quit` | Display closed unexpectedly | Use text builder |
| `ModuleNotFoundError: No module named 'marl_construction'` | Package not installed | `pip install -e .` |
| `CUDA out of memory` | GPU memory full | Reduce batch size or use CPU |
| `ValueError: empty range for randrange()` | Environment too small | Use larger site dimensions |
| `ImportError: cannot import name 'SummaryWriter'` | Wrong tensorboard import | Fixed in current version |

## FAQ

**Q: Can I run without a GPU?**
A: Yes! Everything works on CPU, just slower training.

**Q: Can I modify the environment?**
A: Yes! Edit `construction_env.py` or create custom environments.

**Q: How do I create custom scenarios?**
A: Use the site builder tools to design your own construction challenges.

**Q: Can I train on multiple scenarios?**
A: Yes! Save different sites and train agents on each one.

**Q: How do I visualize training progress?**
A: Use TensorBoard: `tensorboard --logdir logs/`

For more help, check the examples directory or create an issue on GitHub.