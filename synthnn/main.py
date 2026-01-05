"""
Package entry-point used by the `synthnn-demo` console_script.

This mirrors the repository-level `main.py` launcher, but uses package imports so
it works when SynthNN is installed as a library.
"""

from __future__ import annotations

import argparse
import sys


def run_music_demo() -> None:
    print("Launching Modal Music Generation Demonstration...")
    from demos.demo_music_generation import main as music_main

    music_main()


def run_basic_examples() -> None:
    print("Running Basic Usage Examples...")
    from examples.basic_usage import visualize_results

    visualize_results()


def run_accelerated_demo() -> None:
    print("Launching Accelerated Music Generation Demonstration...")
    from demos.demo_accelerated_music import demonstrate_accelerated_music_generation

    demonstrate_accelerated_music_generation()


def run_interactive_shell() -> None:
    print("Launching SynthNN Interactive Shell...")
    print("-" * 50)
    print("Available imports:")
    print("  from synthnn.core import ResonantNode, ResonantNetwork")
    print("  from synthnn.core import SignalProcessor, UniversalPatternCodec")
    print("-" * 50)

    from synthnn.core import ResonantNetwork, ResonantNode, SignalProcessor, UniversalPatternCodec
    import numpy as np  # noqa: F401

    import code

    code.interact(local=locals())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="SynthNN - Synthetic Resonant Neural Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  synthnn-demo --music        # Run music generation demo
  synthnn-demo --examples     # Run basic usage examples
  synthnn-demo --accelerated  # Run accelerated music generation demo
  synthnn-demo --shell        # Launch interactive shell
        """,
    )

    parser.add_argument("--music", action="store_true", help="Run the music generation demonstration")
    parser.add_argument("--examples", action="store_true", help="Run basic usage examples")
    parser.add_argument("--accelerated", action="store_true", help="Run the accelerated music generation demonstration")
    parser.add_argument("--shell", action="store_true", help="Launch interactive Python shell")

    args = parser.parse_args(argv)

    if not any(vars(args).values()):
        parser.print_help()
        print("\nNo option selected. Use --help for more information.")
        return

    try:
        if args.music:
            run_music_demo()
        elif args.examples:
            run_basic_examples()
        elif args.accelerated:
            run_accelerated_demo()
        elif args.shell:
            run_interactive_shell()
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure all dependencies are installed and optional extras are available.")
        sys.exit(1)


if __name__ == "__main__":
    main()

