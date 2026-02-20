"""Response formatter for displaying events by type."""

from typing import TextIO


class ResponseFormatter:
    """Formats and displays chat events organized by type."""

    def __init__(self, output: TextIO, show_thinking: bool = False):
        """Initialize the formatter.

        Parameters
        ----------
        output
            File-like object to write output to (default: stdout).
        show_thinking
            Whether to display thinking events.
        """
        self.output = output
        self.show_thinking = show_thinking
        self.content_buffer: list[str] = []
        self.current_tool: dict | None = None
        self.content_started = False

    def handle_event(self, event: dict) -> None:
        """Handle a single event and display it appropriately.

        Parameters
        ----------
        event
            Parsed JSON event from the API.
        """
        event_type = event.get("type")

        if event_type == "queued":
            position = event.get("position", "?")
            self._print(f"⏳ Queued (position: {position})")

        elif event_type == "thinking":
            if self.show_thinking:
                content = event.get("content", "")
                self._print(f"\nThinking: {content}\n")

        elif event_type == "content":
            content = event.get("content", "")
            self.content_buffer.append(content)
            # Show "Response:" header before first content
            if not self.content_started:
                self._print("\nResponse:\n")
                self.content_started = True
            # Stream content as it arrives
            self.output.write(content)
            self.output.flush()

        elif event_type == "tool_call":
            self._handle_tool_call(event)

        elif event_type == "error":
            message = event.get("message", "Unknown error")
            code = event.get("code", "UNKNOWN")
            self._print(f"\n❌ Error [{code}]: {message}\n")

        else:
            # Handle unknown event types gracefully
            logger = __import__("logging").getLogger(__name__)
            logger.debug(f"Unknown event type: {event_type}, event: {event}")

    def _handle_tool_call(self, event: dict) -> None:
        """Handle a tool call event."""
        name = event.get("name", "unknown")
        status = event.get("status", "unknown")
        arguments = event.get("arguments")
        result = event.get("result")

        if status == "started":
            self.current_tool = {"name": name, "arguments": arguments}
            args_str = self._format_arguments(arguments) if arguments else ""
            self._print(f"\n🔧 Tool: {name}{args_str}\n")

        elif status == "completed":
            result_str = self._format_result(result) if result else ""
            self._print(f"✅ Tool {name} completed{result_str}\n")
            self.current_tool = None

        elif status == "error":
            error_str = self._format_result(result) if result else ""
            self._print(f"❌ Tool {name} failed{error_str}\n")
            self.current_tool = None

    def _format_arguments(self, arguments: dict | None) -> str:
        """Format tool arguments for display."""
        if not arguments:
            return ""
        # Show a summary of arguments
        args_str = ", ".join(f"{k}={v}" for k, v in list(arguments.items())[:3])
        if len(arguments) > 3:
            args_str += "..."
        return f"({args_str})"

    def _format_result(self, result: str | None) -> str:
        """Format tool result for display."""
        if not result:
            return ""
        # Truncate long results
        max_len = 100
        if len(result) > max_len:
            return f": {result[:max_len]}..."
        return f": {result}"

    def finish_response(self) -> None:
        """Finish displaying a response (flush content buffer)."""
        if self.content_buffer:
            # Content was already streamed, just add a newline
            self._print("\n")
            self.content_buffer.clear()
        self.content_started = False

    def _print(self, text: str) -> None:
        """Print text to output."""
        self.output.write(text)
        self.output.flush()
