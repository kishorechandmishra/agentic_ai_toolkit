"""
Agentic Dev Toolkit - Prototype (updated)
File: agentic_dev_toolkit.py
Purpose: Minimal prototype of an agentic developer assistant that:
 - captures human intent
 - asks an LLM for a structured plan (JSON) **or** loads a plan from a local file when the `openai` package is unavailable
 - executes safe, sandboxed steps (dry-run by default)
 - runs tests, lints, and produces a git diff / PR template

This file has been hardened to work even in sandboxes where the `openai` Python
package is not installed. If `openai` is missing, you can pass `--plan-file` to
supply a JSON plan, or `--mock-plan` to use a small built-in example plan.

NOTES / Requirements:
 - Python 3.10+
 - Optional: `pip install openai pydantic typer GitPython`
 - If you want real LLM calls, set OPENAI_API_KEY and install `openai`.
 - This is still a prototype. Review all diffs before pushing.

Usage examples:
  # interactive intent capture + LLM (if available)
  python agentic_dev_toolkit.py --interactive

  # run with an explicit intent (dry-run). If openai isn't installed, use --mock-plan or --plan-file
  python agentic_dev_toolkit.py --intent "Add rate-limiting to streaming API" --dry-run --mock-plan

  # run with a local plan file (bypasses LLM)
  python agentic_dev_toolkit.py --plan-file ./example_plan.json --dry-run

  # execute for real (dangerous):
  python agentic_dev_toolkit.py --intent "Fix bug in token reuse" --apply

Design principles encoded:
 - LLM must return a JSON plan with discrete steps
 - Each step has a type (shell, python, git, test, manual)
 - The tool executes only allowed step types
 - Default is dry-run: prints commands and diffs but doesn't push

"""
from __future__ import annotations
import os
import json
import shlex
import subprocess
import tempfile
from typing import List, Dict, Any, Optional

# Defensive imports: provide graceful fallback when packages are missing
try:
    import typer
    _TYPER_AVAILABLE = True
except Exception:
    _TYPER_AVAILABLE = False

    # ==============================
    # Medium-level Typer Shim
    # ==============================
    import argparse
    class _Option:
        def __init__(self, default=None, help=None, **kwargs):
            self.default = default
            self.help = help

    class _Argument:
        def __init__(self, default=None, help=None, **kwargs):
            self.default = default
            self.help = help

    class _Prompt:
        def __call__(self, text):
            return input(text + " ")

    class _TyperShim:
        def __init__(self):
            self._commands = []

        def Option(self, default=None, help=None, **kwargs):
            return _Option(default=default, help=help)

        def Argument(self, default=None, help=None, **kwargs):
            return _Argument(default=default, help=help)

        def prompt(self, text):
            return _Prompt()(text)

        def command(self):
            def wrapper(func):
                self._commands.append(func)
                return func
            return wrapper

        def __call__(self):
            # Use argparse to parse common flags
            parser = argparse.ArgumentParser()
            parser.add_argument('--intent')
            parser.add_argument('--plan-file')
            parser.add_argument('--mock-plan', action='store_true')
            parser.add_argument('--dry-run', action='store_true')
            parser.add_argument('--apply-changes', action='store_true')
            parser.add_argument('--repo', default='.')
            args = parser.parse_args()
            if not self._commands:
                return
            # Call the single command
            return self._commands[0](
                intent=args.intent,
                plan_file=args.plan_file,
                mock_plan=args.mock_plan,
                dry_run=args.dry_run,
                apply_changes=args.apply_changes,
                repo=args.repo,
                interactive=False,
            )

    typer = _TyperShim()

# Initialize app
if _TYPER_AVAILABLE:
    app = typer.Typer()
else:
    app = typer  # Typer shim instance

# --- Configuration ---
OPENAI_MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")  # change as desired
DRY_RUN_DEFAULT = True
ALLOWED_STEP_TYPES = {"shell", "git", "test", "python", "manual"}

# --- Ensure BaseModel, Field, Repo, openai fallbacks ---
try:
    from pydantic import BaseModel, Field
except Exception:
    class BaseModel:
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)
        def dict(self): return self.__dict__
        def json(self, **kwargs): return json.dumps(self.__dict__)
    def Field(default=None, **kwargs): return default

try:
    from git import Repo
    GITPY_AVAILABLE = True
except Exception:
    Repo = None
    GITPY_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_AVAILABLE = False

# --- Plan schema ---
class PlanStep(BaseModel):
    id: str
    type: str
    description: str
    command: Optional[str] = None
    files_touched: List[str] = Field(default_factory=list)
    risk: str = "low"  # low/medium/high

class Plan(BaseModel):
    summary: str
    steps: List[PlanStep]

# --- LLM helpers ---

def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # naive strip
        parts = text.split("\n", 1)
        if len(parts) > 1:
            text = parts[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text


def call_llm_for_plan(intent: str) -> Plan:
    """Call the LLM and return a structured Plan. If the openai package is
    unavailable this function will raise a descriptive error and the CLI
    can fall back to --plan-file or --mock-plan.
    """
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI client not available in this environment. Use --plan-file or --mock-plan to provide a plan without calling the API.")

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required for LLM calls")

    system_prompt = (
        "You are a senior engineer assistant. Produce a JSON plan with fields: summary and steps. "
        "Each step must include id, type (shell|git|test|python|manual), description, command (optional), files_touched (list), risk (low|medium|high). "
        "Output ONLY valid JSON. Do not include any explanatory text outside the JSON."
    )
    user_prompt = f"Intent:\n{intent}\n\nConstraints: keep changes minimal and safe. Return a plan JSON."

    # Support both ChatCompletion and the newer Responses API surfaces if available
    if hasattr(openai, "ChatCompletion"):
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=800,
        )
        text = response["choices"][0]["message"]["content"].strip()
    else:
        # fallback generic call (may not exist in older/newer SDKs)
        response = openai.Completion.create(
            model=OPENAI_MODEL,
            prompt=system_prompt + "\n\n" + user_prompt,
            temperature=0.0,
            max_tokens=800,
        )
        text = response["choices"][0]["text"].strip()

    text = _strip_code_fence(text)
    plan_json = json.loads(text)
    # pydantic BaseModel shim accepts keyword init; if using real pydantic, this validates.
    plan = Plan(**plan_json)

    # validate step types
    for s in plan.steps:
        if s.type not in ALLOWED_STEP_TYPES:
            raise ValueError(f"Disallowed step type in plan: {s.type}")
    return plan

# --- Execution helpers ---

def run_shell(command: str, cwd: Optional[str] = None, capture: bool = True) -> Dict[str, Any]:
    if not command:
        return {"returncode": 0, "stdout": "", "stderr": ""}
    args = shlex.split(command)
    res = subprocess.run(args, cwd=cwd, capture_output=capture, text=True)
    return {"returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr}


def show_git_diff(repo_path: str) -> str:
    if not GITPY_AVAILABLE or Repo is None:
        return "(gitpython not available; cannot show diff)"
    repo = Repo(repo_path)
    dif = repo.git.execute(["git", "--no-pager", "diff", "--staged"])
    if not dif:
        dif = repo.git.execute(["git", "--no-pager", "diff"])
    return dif


def execute_plan(plan: Plan, repo_path: str = ".", dry_run: bool = True) -> Dict[str, Any]:
    results: Dict[str, Any] = {"steps": []}
    if GITPY_AVAILABLE and Repo is not None:
        try:
            repo = Repo(repo_path)
        except Exception:
            repo = None
    else:
        repo = None

    for step in plan.steps:
        res: Dict[str, Any] = {"id": step.id, "type": step.type, "description": step.description, "ok": False}
        print(f"\n[STEP] {step.id} — {step.type} — risk={step.risk}: {step.description}")
        if step.command:
            print(f"COMMAND: {step.command}")

        if step.type == "manual":
            res["note"] = "Manual step — human review required"
            res["ok"] = False
            results["steps"].append(res)
            continue

        if dry_run:
            # In dry-run we'll print but not apply destructive steps.
            res["dry_run"] = True
            if step.type == "git":
                print("(dry-run) git step — showing affected files:", step.files_touched)
                res["ok"] = True
            elif step.type == "test":
                print("(dry-run) would run test command:", step.command)
                res["ok"] = True
            else:
                print("(dry-run) would run:", step.command)
                res["ok"] = True
            results["steps"].append(res)
            continue

        # Non-dry-run execution
        try:
            if step.type == "shell":
                r = run_shell(step.command or "", cwd=repo_path)
                res.update(r)
                res["ok"] = r["returncode"] == 0

            elif step.type == "python":
                # For safety, run python commands in a temporary process
                r = run_shell(f"python -c {shlex.quote(step.command or '')}", cwd=repo_path)
                res.update(r)
                res["ok"] = r["returncode"] == 0

            elif step.type == "test":
                r = run_shell(step.command or "pytest -q", cwd=repo_path)
                res.update(r)
                res["ok"] = r["returncode"] == 0

            elif step.type == "git":
                r = run_shell(step.command or "", cwd=repo_path)
                res.update(r)
                res["ok"] = r["returncode"] == 0

            else:
                res["error"] = "Unknown step type"
                res["ok"] = False

        except Exception as e:
            res["error"] = str(e)
            res["ok"] = False

        results["steps"].append(res)

    # at end show diff
    try:
        diff = show_git_diff(repo_path)
        results["git_diff"] = diff
    except Exception as e:
        results["git_diff_error"] = str(e)

    return results

# --- Utilities ---

def create_pr_template(plan: Plan) -> str:
    title = f"[AI] {plan.summary[:60]}"
    body = """
This PR was generated by the agentic_dev_toolkit prototype.

Plan summary:

"""
    body += plan.summary + "\n\nSteps:\n"
    for s in plan.steps:
        body += f"- {s.id}: {s.description} (risk={s.risk})\n"
    body += "\nAgent transcript omitted. Review carefully."
    return f"{title}\n\n{body}"

# --- CLI ---

@app.command()
def run(
    intent: Optional[str] = typer.Option(None, help="Intent or feature request"),
    interactive: bool = typer.Option(False, help="Open interactive prompt to capture intent"),
    repo: str = typer.Option('.', help='Path to repository'),
    dry_run: bool = typer.Option(DRY_RUN_DEFAULT, help='Dry run (default True)'),
    apply_changes: bool = typer.Option(False, help='Apply changes (dangerous)'),
    plan_file: Optional[str] = typer.Option(None, help='Path to a JSON plan file (bypass LLM)'),
    mock_plan: bool = typer.Option(False, help='Use an embedded mock plan instead of calling the LLM')
):
    """Main entrypoint. If openai isn't available, prefer --plan-file or --mock-plan."""
    if interactive and not intent:
        # If openai is not available, interactive mode will still capture intent but require --mock-plan or --plan-file
        intent = typer.prompt("Describe what you want the agent to do (feature / bug / refactor)")

    if not intent and not plan_file and not mock_plan:
        # Auto-enter interactive mode (Option 2)
        try:
            intent = typer.prompt("Describe what you want the agent to do (feature / bug / refactor)")
        except Exception:
            typer.echo("Interactive mode unavailable. Use --intent, --plan-file, or --mock-plan.")
            raise typer.Exit(code=1)

    typer.echo(f"Captured intent: {intent}")

    # Determine plan source
    plan: Optional[Plan] = None
    if plan_file:
        typer.echo(f"Loading plan from file: {plan_file}")
        with open(plan_file, 'r') as f:
            plan_json = json.load(f)
        plan = Plan(**plan_json)
    elif mock_plan:
        typer.echo("Using built-in mock plan")
        plan = Plan(**{
            "summary": "Mock: add a simple hello.txt file",
            "steps": [
                {"id": "1", "type": "shell", "description": "Create hello.txt", "command": "bash -c 'echo hello > hello.txt'", "files_touched": ["hello.txt"], "risk": "low"},
                {"id": "2", "type": "git", "description": "Stage and commit hello.txt", "command": "git add hello.txt && git commit -m 'Add hello.txt' || true", "files_touched": ["hello.txt"], "risk": "low"}
            ]
        })
    else:
        # call the LLM (may raise if openai not available)
        typer.echo("Requesting plan from LLM...")
        plan = call_llm_for_plan(intent or "")

    typer.echo("Plan received:")
    try:
        typer.echo(plan.json(indent=2))
    except Exception:
        typer.echo(str(plan))

    # safety: only allow apply_changes if explicitly requested
    run_dry = dry_run and not apply_changes
    if apply_changes and dry_run:
        typer.echo("NOTE: --apply requested; overriding --dry-run")
        run_dry = False

    result = execute_plan(plan, repo_path=repo, dry_run=run_dry)

    typer.echo("\nExecution summary:")
    typer.echo(json.dumps(result, indent=2)[:1000])

    # Construct PR template and show diff
    pr_text = create_pr_template(plan)
    typer.echo('\n=== PR TEMPLATE ===')
    typer.echo(pr_text)

    typer.echo('\n=== GIT DIFF ===')
    typer.echo(result.get('git_diff', '(no diff)')[:2000])

    typer.echo('\nNext steps: review the diff above, run tests locally, and push with your normal flow.')


# --- Unit tests ---
if __name__ == "__main__":
    # If RUN_UNIT_TESTS env var is set, run a small self-test suite and exit.
    import sys
    if os.getenv("RUN_UNIT_TESTS") == "1":
        import unittest

        class ToolkitTests(unittest.TestCase):
            def test_create_pr_template_and_execute_dry(self):
                plan = Plan(**{
                    "summary": "Test plan",
                    "steps": [
                        {"id": "s1", "type": "shell", "description": "echo hi", "command": "echo hi", "files_touched": [], "risk": "low"},
                        {"id": "s2", "type": "test", "description": "run tests", "command": "pytest -q", "files_touched": [], "risk": "low"},
                        {"id": "s3", "type": "manual", "description": "Manual review", "files_touched": [], "risk": "low"}
                    ]
                })
                pr = create_pr_template(plan)
                self.assertIsInstance(pr, str)
                res = execute_plan(plan, repo_path='.', dry_run=True)
                self.assertIn('steps', res)
                self.assertEqual(len(res['steps']), 3)
                self.assertTrue(all('id' in s for s in res['steps']))

            def test_mock_plan_loading(self):
                # ensure mock plan structure is valid
                mock = {
                    "summary": "Mock",
                    "steps": [{"id":"1","type":"shell","description":"noop","command":"echo ok","files_touched":[],"risk":"low"}]
                }
                plan = Plan(**mock)
                self.assertEqual(plan.summary, "Mock")

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ToolkitTests)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 2)

    # Always use shim CLI to avoid Typer SystemExit issues
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mock-plan', action='store_true')
parser.add_argument('--plan-file')
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--intent')
parser.add_argument('--repo', default='.')
parser.add_argument('--apply-changes', action='store_true')
args = parser.parse_args()
if args.plan_file:
    with open(args.plan_file) as f:
        plan = Plan(**json.load(f))
elif args.mock_plan:
    plan = Plan(**{
        "summary": "Mock",
        "steps": [{"id":"1","type":"shell","description":"noop","command":"echo ok","files_touched":[],"risk":"low"}]
    })
elif args.intent:
    plan = Plan(**{
        "summary": f"Intent: {args.intent}",
        "steps": []
    })
else:
    # No interactive fallback to avoid hangs
    print("No plan provided. Use --mock-plan, --plan-file, or --intent.")
    raise SystemExit(0)  # clean exit, no callable error

print('Running dry-run...')
execute_plan(plan, repo_path=args.repo, dry_run=not args.apply_changes)
