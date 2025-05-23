#!/usr/bin/env python3
"""
Main entry point for the SynthNN project.

This script provides a launcher for various demonstrations of the framework.
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_music_demo():
    """Run the music generation demonstration."""
    print("Launching Modal Music Generation Demonstration...")
    try:
        from demo_music_generation import main as music_main
        music_main()
    except ImportError as e:
        print(f"Error: Could not import music demo - {e}")
        print("Make sure all dependencies are installed.")
        sys.exit(1)


def run_basic_examples():
    """Run basic usage examples."""
    print("Running Basic Usage Examples...")
    try:
        from examples.basic_usage import visualize_results
        visualize_results()
    except ImportError as e:
        print(f"Error: Could not import basic examples - {e}")
        sys.exit(1)


def run_interactive_shell():
    """Launch an interactive Python shell with SynthNN imported."""
    print("Launching SynthNN Interactive Shell...")
    print("-" * 50)
    print("Available imports:")
    print("  from synthnn.core import ResonantNode, ResonantNetwork")
    print("  from synthnn.core import SignalProcessor, UniversalPatternCodec")
    print("-" * 50)
    
    # Import modules for the interactive session
    from synthnn.core import (
        ResonantNode, 
        ResonantNetwork,
        SignalProcessor,
        UniversalPatternCodec
    )
    import numpy as np
    
    # Launch interactive shell
    import code
    code.interact(local=locals())


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="SynthNN - Synthetic Resonant Neural Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --music        # Run music generation demo
  python main.py --examples     # Run basic usage examples  
  python main.py --shell        # Launch interactive shell
        """
    )
    
    parser.add_argument(
        '--music', 
        action='store_true',
        help='Run the music generation demonstration'
    )
    parser.add_argument(
        '--examples',
        action='store_true', 
        help='Run basic usage examples'
    )
    parser.add_argument(
        '--shell',
        action='store_true',
        help='Launch interactive Python shell'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        print("\nNo option selected. Use --help for more information.")
        sys.exit(0)
    
    # Run selected option
    if args.music:
        run_music_demo()
    elif args.examples:
        run_basic_examples()
    elif args.shell:
        run_interactive_shell()


if __name__ == "__main__":
    main()
