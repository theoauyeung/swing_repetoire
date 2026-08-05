"""Watches the project folder and commits to git on its own once edits stop. Convenience only,
nothing to do with the analysis.

Usage:
    .venv\Scripts\python.exe src\commit.py

Polls every POLL_INTERVAL seconds. Once changes are detected, waits
DEBOUNCE_SECS of quiet (no new changes) before committing. Stop with Ctrl+C.
"""

import subprocess
import time
import sys
from pathlib import Path

POLL_INTERVAL = 2    # seconds between git status checks
DEBOUNCE_SECS = 30   # seconds of quiet before committing — long enough for plot generation

# Repo root, not src/ — so `git add .` stages the whole working tree, not just this dir.
_toplevel_result = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True, text=True, cwd=Path(__file__).parent,
)
ROOT = Path(_toplevel_result.stdout.strip())


def _git(*args):
    return subprocess.run(
        ["git"] + list(args),
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def _has_changes():
    status_output = _git("status", "--porcelain").stdout.strip()
    return bool(status_output)


def _build_commit_label(staged_files):
    """Build a short human-readable label from the list of staged file names."""
    label = ", ".join(staged_files[:3])
    if len(staged_files) > 3:
        label += f" (+{len(staged_files) - 3} more)"
    return label


def _pull_if_behind():
    """Fetch origin and rebase if remote has commits we don't have locally.

    Stashes any uncommitted local changes first, rebases onto the remote branch,
    then pops the stash so local edits survive. Returns True on success.
    """
    _git("fetch", "origin")
    current_branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()

    count_result = _git("rev-list", "--count", f"HEAD..origin/{current_branch}")
    remote_is_absent_or_current = (
        count_result.returncode != 0
        or count_result.stdout.strip() in ("", "0")
    )
    if remote_is_absent_or_current:
        return True  # remote ref doesn't exist yet or already up to date

    new_commit_count = count_result.stdout.strip()
    print(f"  remote has {new_commit_count} new commit(s) on {current_branch}, pulling first...")

    # Stash local changes (including untracked) so rebase has a clean tree
    stash_result = _git("stash", "push", "-u", "-m", "watch_commit: pre-pull")
    did_stash = "Saved" in stash_result.stdout

    rebase_result = _git("rebase", f"origin/{current_branch}")
    if rebase_result.returncode != 0:
        print(f"  rebase failed: {rebase_result.stderr.strip()}", file=sys.stderr)
        _git("rebase", "--abort")
        if did_stash:
            _git("stash", "pop")
        return False

    if did_stash:
        pop_result = _git("stash", "pop")
        if pop_result.returncode != 0:
            print(f"  stash pop conflict after pull: {pop_result.stderr.strip()}", file=sys.stderr)
            return False

    print(f"  pulled {new_commit_count} remote commit(s) successfully.")
    return True


def _commit():
    if not _pull_if_behind():
        return

    _git("add", ".")
    staged_output = _git("diff", "--cached", "--name-only").stdout.strip()
    staged_files = staged_output.splitlines()
    if not staged_files:
        return

    commit_label = _build_commit_label(staged_files)
    commit_message = f"auto: {commit_label}"

    commit_result = _git("commit", "-m", commit_message)
    if commit_result.returncode != 0:
        print(f"  commit failed: {commit_result.stderr.strip()}", file=sys.stderr)
        return

    short_hash = _git("rev-parse", "--short", "HEAD").stdout.strip()
    print(f"  [{short_hash}] {commit_message} — pushing...", end="", flush=True)
    push_result = _git("push", "origin", "HEAD")
    if push_result.returncode == 0:
        print(" done.")
    else:
        print(f" push failed: {push_result.stderr.strip()}", file=sys.stderr)


def main():
    print(f"Watching for changes (poll={POLL_INTERVAL}s, debounce={DEBOUNCE_SECS}s) — Ctrl+C to stop.")
    last_change_time = None  # timestamp of when we first detected the current batch of changes

    while True:
        try:
            if _has_changes():
                current_time = time.time()
                if last_change_time is None:
                    # First detection of this batch — start the debounce clock
                    last_change_time = current_time
                    print("  changes detected, waiting for quiet...")
                elif current_time - last_change_time >= DEBOUNCE_SECS:
                    # Quiet long enough — commit and reset
                    _commit()
                    last_change_time = None
            else:
                last_change_time = None

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
