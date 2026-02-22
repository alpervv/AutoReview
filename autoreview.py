"""
AutoReview — 3-stage AI review pipeline using Claude and Codex CLIs.
Read-only analysis of a target codebase directory.
"""

import asyncio
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBPROCESS_TIMEOUT = 300  # seconds

CROSS_REVIEW_CONTEXT = (
    "Here is the report that we received from a company. Review the report generated and compare "
    "with your findings to create a more accurate final report. In the report, do not include "
    "comparisons, only include findings that are more accurate."
)

ARBITRATION_CONTEXT = (
    "Here are 2 reports generated for the issue below. Determine which report is more accurate.\n"
    "Return only a single character: 1 or 2.\n"
    "Do not include any explanation, words, punctuation, or markdown.\n\n"
    "{user_prompt}\n\n"
    "--- Report 1 (Claude) ---\n"
    "{claude_report_2}\n\n"
    "--- Report 2 (Codex) ---\n"
    "{codex_report_2}"
)

console = Console()

# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _resolve_exe(name: str) -> list[str]:
    """Return the command prefix needed to run a CLI tool on this platform.
    On Windows, .cmd/.bat files must be invoked via 'cmd /c'."""
    if sys.platform == "win32":
        # npm global bins often include multiple shims (e.g. `codex`, `codex.cmd`, `codex.ps1`).
        # `shutil.which("codex")` may return the extensionless shim first, which cannot be
        # executed directly by CreateProcess and raises WinError 193.
        path = None
        for candidate in (f"{name}.cmd", f"{name}.bat", f"{name}.exe", name):
            path = shutil.which(candidate)
            if path:
                break
        path = path or name
    else:
        path = shutil.which(name) or name

    if sys.platform == "win32" and path.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", path]
    return [path]


async def run_claude(prompt: str, cwd: Path) -> str:
    """Invoke the claude CLI and return the final response text (reasoning-free)."""
    cmd = _resolve_exe("claude") + ["--output-format", "json", "-p", prompt]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=SUBPROCESS_TIMEOUT)
    except asyncio.TimeoutError:
        raise RuntimeError(f"Stage timed out after {SUBPROCESS_TIMEOUT}s (claude)")

    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        detail = stderr_text or stdout_text or "(no output)"
        raise RuntimeError(f"claude exited with code {proc.returncode}: {detail}")

    if not stdout_text:
        raise RuntimeError("claude returned an empty response")

    # Parse JSON and extract the result field
    try:
        data = json.loads(stdout_text)
        result = data.get("result") or data.get("content") or data.get("text")
        if result is None:
            # Try first item if it's a list
            if isinstance(data, list) and data:
                first = data[0]
                result = first.get("result") or first.get("content") or first.get("text")
        if result is not None:
            return str(result).strip()
        # Fallback: return raw stdout
        return stdout_text
    except (json.JSONDecodeError, AttributeError):
        # Fallback to raw stdout
        return stdout_text


async def run_codex(
    prompt: str,
    cwd: Path,
    session_id: str | None = None,
) -> tuple[str, str | None]:
    """
    Invoke the codex CLI and return (response_text, session_id_if_found).
    If session_id is provided, continues that session.
    """
    # codex-cli >=0.104 uses the non-interactive `exec` subcommand and `--full-auto`.
    # Pass the prompt over stdin (`-`) to avoid Windows cmd.exe command-line length truncation,
    # which can cut off large Stage 2/3 prompts and produce unrelated answers.
    base = _resolve_exe("codex") + ["exec", "--full-auto", "--skip-git-repo-check", "-"]
    if session_id:
        # Newer codex CLI resumes via `exec resume <session_id> <prompt>`.
        cmd = _resolve_exe("codex") + [
            "exec",
            "resume",
            "--full-auto",
            "--skip-git-repo-check",
            session_id,
            "-",
        ]
    else:
        cmd = base

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(prompt.encode("utf-8")),
            timeout=SUBPROCESS_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"Stage timed out after {SUBPROCESS_TIMEOUT}s (codex)")

    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()

    if proc.returncode != 0:
        detail = stderr_text or stdout_text or "(no output)"
        raise RuntimeError(f"codex exited with code {proc.returncode}: {detail}")

    if not stdout_text:
        raise RuntimeError("codex returned an empty response")

    # Extract session ID from output if present
    found_session_id = _extract_codex_session_id(stdout_text)

    # Strip reasoning and get the final response text
    response = strip_reasoning_codex(stdout_text)

    return response, found_session_id


def _extract_codex_session_id(raw: str) -> str | None:
    """Try to find a session ID in codex output."""
    # Common patterns: "session: <id>", "Session ID: <id>", JSON {"session_id": "..."}
    patterns = [
        r'"session[_\-]?id"\s*:\s*"([^"]+)"',
        r'session[_\-]?id[:\s]+([a-zA-Z0-9\-_]+)',
        r'Session[:\s]+([a-zA-Z0-9\-_]{8,})',
    ]
    for pattern in patterns:
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------


def strip_reasoning_codex(raw: str) -> str:
    """Strip <thinking>…</thinking> blocks and extract the last assistant turn."""
    # Remove thinking blocks (dotall)
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL | re.IGNORECASE)

    # Try to extract last assistant content block
    # Patterns like "Assistant:" or "assistant\n" followed by content
    assistant_patterns = [
        r"(?:^|\n)(?:assistant|ASSISTANT)\s*[:\-]\s*([\s\S]+?)(?=\n(?:user|human|system|assistant|USER|HUMAN|SYSTEM|ASSISTANT)\s*[:\-]|$)",
        r"(?:^|\n)(?:assistant|ASSISTANT)\s*\n([\s\S]+?)(?=\n(?:user|human|system|assistant|USER|HUMAN|SYSTEM|ASSISTANT)\s*\n|$)",
    ]

    last_match = None
    for pattern in assistant_patterns:
        for m in re.finditer(pattern, cleaned, re.IGNORECASE):
            last_match = m.group(1).strip()

    if last_match:
        return last_match

    # If no assistant markers, return cleaned text as-is
    return cleaned.strip()


def parse_arbitration_choice(raw: str) -> int:
    """Parse Codex arbitration output and return 1 or 2."""
    text = raw.strip()
    if text in {"1", "2"}:
        return int(text)

    # Tolerate minor format drift like "Report 2" or "2." while keeping the contract strict.
    match = re.search(r"\b([12])\b", text)
    if match:
        return int(match.group(1))

    raise RuntimeError(f"Stage 3 returned invalid arbitration choice: {text!r}")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_cross_review_prompt(other_report: str) -> str:
    """Build the Stage 2 cross-review prompt."""
    return f"{CROSS_REVIEW_CONTEXT}\n\n{other_report}"


def build_arbitration_prompt(
    user_prompt: str,
    claude_report: str,
    codex_report: str,
) -> str:
    """Build the Stage 3 arbitration prompt."""
    return ARBITRATION_CONTEXT.format(
        user_prompt=user_prompt,
        claude_report_2=claude_report,
        codex_report_2=codex_report,
    )


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------


def save_report(cwd: Path, user_prompt: str, supreme_report: str) -> Path:
    """Write review_YYYYMMDD_HHMMSS.md to cwd and return the path."""
    timestamp = datetime.now()
    filename = f"review_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
    output_path = cwd / filename

    content = (
        "# AutoReview Report\n\n"
        f"**Date:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"**Query:** {user_prompt}\n\n"
        "---\n\n"
        "## Supreme Report\n\n"
        f"{supreme_report}\n"
    )

    output_path.write_text(content, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


async def run_pipeline(user_prompt: str, cwd: Path) -> None:
    """Orchestrate the full 3-stage review pipeline."""

    # --- Stage 1 — Parallel initial analysis ---
    with _spinner("Stage 1/3 — Parallel analysis (Claude + Codex)..."):
        stage1_results = await asyncio.gather(
            run_claude(user_prompt, cwd),
            run_codex(user_prompt, cwd),
            return_exceptions=True,
        )

    claude_result_1 = stage1_results[0]
    codex_result_1 = stage1_results[1]

    if isinstance(claude_result_1, Exception):
        raise RuntimeError(f"Stage 1 Claude failed: {claude_result_1}") from claude_result_1
    if isinstance(codex_result_1, Exception):
        raise RuntimeError(f"Stage 1 Codex failed: {codex_result_1}") from codex_result_1

    claude_report_1: str = claude_result_1
    codex_response_1, codex_session_id = codex_result_1  # type: ignore[misc]

    console.print(f"  [dim]Stage 1 complete. Codex session: {codex_session_id or 'none'}[/dim]")

    # --- Stage 2 — Parallel cross-review ---
    with _spinner("Stage 2/3 — Cross-review (Claude reviews Codex, Codex reviews Claude)..."):
        cross_prompt_for_claude = build_cross_review_prompt(codex_response_1)
        cross_prompt_for_codex = build_cross_review_prompt(claude_report_1)

        stage2_results = await asyncio.gather(
            run_claude(cross_prompt_for_claude, cwd),
            run_codex(cross_prompt_for_codex, cwd, session_id=codex_session_id),
            return_exceptions=True,
        )

    claude_result_2 = stage2_results[0]
    codex_result_2 = stage2_results[1]

    if isinstance(claude_result_2, Exception):
        raise RuntimeError(f"Stage 2 Claude failed: {claude_result_2}") from claude_result_2
    if isinstance(codex_result_2, Exception):
        raise RuntimeError(f"Stage 2 Codex failed: {codex_result_2}") from codex_result_2

    claude_report_2: str = claude_result_2
    codex_report_2_text, _ = codex_result_2  # type: ignore[misc]

    # --- Stage 3 — Arbitration ---
    with _spinner("Stage 3/3 — Arbitration (Codex synthesises supreme report)..."):
        arbitration_prompt = build_arbitration_prompt(
            user_prompt=user_prompt,
            claude_report=claude_report_2,
            codex_report=codex_report_2_text,
        )
        arb_result = await run_codex(arbitration_prompt, cwd, session_id=None)

    arbitration_response, _ = arb_result
    winner = parse_arbitration_choice(arbitration_response)
    supreme_report = claude_report_2 if winner == 1 else codex_report_2_text

    if not supreme_report.strip():
        raise RuntimeError("Stage 3 returned an empty supreme report")

    # --- Save ---
    report_path = save_report(cwd, user_prompt, supreme_report)
    console.print(f"\n[bold green]Report saved:[/bold green] {report_path}")


# ---------------------------------------------------------------------------
# Rich spinner context manager
# ---------------------------------------------------------------------------


class _spinner:
    """Simple context manager that shows a rich spinner while work runs."""

    def __init__(self, label: str) -> None:
        self._label = label
        self._live: Live | None = None

    def __enter__(self) -> "_spinner":
        spinner = Spinner("dots", text=Text(self._label, style="bold cyan"))
        self._live = Live(spinner, console=console, transient=True, refresh_per_second=10)
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._live:
            self._live.__exit__(*args)
        console.print(f"  [green]✓[/green] {self._label.rstrip('.')}")


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------


def main() -> None:
    cwd = Path.cwd()

    console.print(
        Panel(
            f"[bold cyan]AutoReview[/bold cyan]\n"
            f"3-stage AI review pipeline  •  Claude Opus 4.6 + Codex 5.3\n\n"
            f"[dim]Analysing:[/dim] [yellow]{cwd}[/yellow]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    while True:
        try:
            query = console.input("\n[bold]Enter your query[/bold] (or [dim]quit[/dim] to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting.[/dim]")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        try:
            asyncio.run(run_pipeline(query, cwd))
        except RuntimeError as exc:
            console.print(f"\n[bold red]Error:[/bold red] {exc}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"\n[bold red]Unexpected error:[/bold red] {exc}")


if __name__ == "__main__":
    main()
