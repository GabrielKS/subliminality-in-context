"""Example script demonstrating that scripts depend on the library.

Run it with uv:

    uv run scripts/run_example.py
"""

from subliminality import greet


def main() -> None:
    print(greet("world"))


if __name__ == "__main__":
    main()
