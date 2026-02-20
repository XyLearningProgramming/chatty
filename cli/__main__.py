"""Entry point for running the CLI as a module."""

import argparse
import asyncio
import sys

from .chatty_cli import main


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive CLI for the Chatty API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )
    parser.add_argument(
        "--api-path",
        type=str,
        default="/api/v1/chatty/chat",
        help="API path (default: /api/v1/chatty/chat)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (shows response headers)",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Show thinking events (default: show)",
    )

    return parser.parse_args()


def cli_entry() -> None:
    """CLI entry point."""
    args = parse_args()

    try:
        asyncio.run(
            main(
                host=args.host,
                port=args.port,
                api_path=args.api_path,
                debug=args.debug,
                show_thinking=args.show_thinking,
            )
        )
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    cli_entry()
