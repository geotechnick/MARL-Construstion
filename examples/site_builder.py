#!/usr/bin/env python3
"""
Interactive Construction Site Builder

This tool allows you to design custom construction scenarios for MARL training.
Create your own layouts with materials, tasks, and agent spawn points.
"""

import argparse
from marl_construction.envs import ConstructionSiteBuilder

def main():
    parser = argparse.ArgumentParser(description="Interactive Construction Site Builder")
    parser.add_argument("--width", type=int, default=15, help="Site width (default: 15)")
    parser.add_argument("--height", type=int, default=15, help="Site height (default: 15)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏗️  MARL Construction Site Builder")
    print("=" * 60)
    print(f"Creating {args.width}x{args.height} construction site...")
    print()
    
    # Create and run the builder
    builder = ConstructionSiteBuilder(width=args.width, height=args.height)
    builder.run()

if __name__ == "__main__":
    main()