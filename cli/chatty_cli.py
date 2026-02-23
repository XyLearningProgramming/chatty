"""Main CLI loop for interactive chat."""

import logging
import sys
from typing import TextIO

from .client import ChatAPIClient
from .config import CLIConfig
from .formatter import ResponseFormatter

logger = logging.getLogger(__name__)


class ChattyCLI:
    """Interactive CLI for the Chatty API."""

    def __init__(
        self,
        config: CLIConfig,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
        show_thinking: bool = False,
    ):
        """Initialize the CLI.

        Parameters
        ----------
        config
            CLI configuration.
        input_stream
            Input stream for user input (default: stdin).
        output_stream
            Output stream for responses (default: stdout).
        show_thinking
            Whether to show thinking events.
        """
        self.config = config
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.show_thinking = show_thinking
        self.client = ChatAPIClient(config)
        self.conversation_id: str | None = None

    async def run(self) -> None:
        """Run the interactive CLI loop."""
        try:
            self._print_welcome()
            while True:
                try:
                    query = self._get_user_input()
                    if not query:
                        continue

                    if query.strip().lower() in ("exit", "quit", "q"):
                        self._print("Goodbye!\n")
                        break

                    await self._process_query(query)

                except KeyboardInterrupt:
                    self._print("\n\nInterrupted. Use 'exit' or 'quit' to exit.\n")
                except EOFError:
                    self._print("\nGoodbye!\n")
                    break
        finally:
            await self.client.close()

    async def _process_query(self, query: str) -> None:
        """Process a single user query."""
        formatter = ResponseFormatter(self.output_stream, self.show_thinking)

        try:
            async for event in self.client.chat(query, self.conversation_id):
                # Handle metadata events (conversation_id from headers)
                if event.get("type") == "_metadata":
                    new_conversation_id = event.get("conversation_id")
                    if new_conversation_id:
                        self.conversation_id = new_conversation_id
                    continue

                formatter.handle_event(event)

            formatter.finish_response()
            self._print("\n")

        except Exception as e:
            logger.exception("Error processing query")
            self._print(f"\n❌ Error: {str(e)}\n\n")

    def _get_user_input(self) -> str:
        """Get user input from the input stream."""
        try:
            self._print("> ")
            line = self.input_stream.readline()
            if not line:
                raise EOFError
            return line.rstrip("\n\r")
        except (EOFError, KeyboardInterrupt):
            raise

    def _print_welcome(self) -> None:
        """Print welcome message."""
        self._print("Chatty CLI - Interactive Chat Interface\n")
        self._print(f"Connected to: {self.config.chat_url}\n")
        self._print(
            "Type your message and press Enter. Type 'exit' or 'quit' to exit.\n\n"
        )

    def _print(self, text: str) -> None:
        """Print text to output stream."""
        self.output_stream.write(text)
        self.output_stream.flush()


async def main(
    host: str = "localhost",
    port: int = 8080,
    api_path: str = "/api/v1/chatty/chat",
    debug: bool = False,
    show_thinking: bool = False,
) -> None:
    """Main entry point for the CLI.

    Parameters
    ----------
    host
        Server host.
    port
        Server port.
    api_path
        API path.
    debug
        Enable debug logging.
    show_thinking
        Show thinking events.
    """
    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    # Create config
    config = CLIConfig(host=host, port=port, api_path=api_path)

    # Run CLI
    cli = ChattyCLI(config, show_thinking=show_thinking)
    await cli.run()
