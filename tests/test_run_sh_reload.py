"""run.sh must not read past its top-level `case` once a command has run.

Bash reads a script incrementally: after the `case ... esac` it reads the next line from
the file's old byte offset. A `git pull` that changes run.sh while a long command (a
dist-worker pipeline) runs makes that offset land mid-text, so bash fails with a syntax
error (exit 2) after the command itself succeeded. `esac; exit` makes bash parse the exit
together with the case, so nothing after it is ever read.
"""
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def test_text_appended_after_the_case_is_never_read(tmp_path):
    script = tmp_path / "run.sh"
    shutil.copy(REPO_ROOT / "run.sh", script)
    # Stands in for the old offset landing inside a later version's quoted text.
    with script.open("a") as f:
        f.write("echo \"unterminated: it's\n")
    proc = subprocess.run(
        ["bash", str(script), "help"], cwd=tmp_path, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "unexpected EOF" not in proc.stderr
