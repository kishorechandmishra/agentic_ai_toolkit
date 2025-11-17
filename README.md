Agentic Dev Toolkit

A minimal, sandbox-friendly prototype of an agentic developer assistant.
It captures intent, generates or loads a structured plan, executes steps safely
(dry-run by default), and produces diffs + PR templates.

This toolkit is designed to run even when common dependencies—openai, typer,
pydantic, or gitpython—are not available.

✨ Features

Intent Capture (from CLI flags or local plan file)

Optional LLM Planning (falls back to local plan if OpenAI is unavailable)

Deterministic Step Execution: shell, python, git, test, manual

Dry-Run Safety: no file changes unless explicitly allowed

Automatic PR Template Generation

Graceful Fallbacks when dependencies are missing

Basic Unit Tests included

🚀 Usage
Mock plan (safe, recommended in restricted environments)
python agentic_dev_toolkit.py --mock-plan

Run with an explicit intent
python agentic_dev_toolkit.py --intent "Refactor token parser"

Load a JSON plan file
python agentic_dev_toolkit.py --plan-file ./plan.json

Execute changes (dangerous)
python agentic_dev_toolkit.py --intent "Fix cache invalidation" --apply-changes

📦 Installation (optional)

Install full dependencies:

pip install openai pydantic typer GitPython


Without these, the program runs in fallback mode using internal shims.

To enable LLM planning:

export OPENAI_API_KEY="your_api_key_here"

🧠 How It Works
1. Plan Generation

From LLM (if available)

From --mock-plan

From --plan-file

2. Execution

Dry-run by default

Prints all actions instead of performing them (safe mode)

3. PR Output

Generates:

A clean PR description

Git diff (if GitPython is available)

🧪 Running Tests

Run built-in tests:

RUN_UNIT_TESTS=1 python agentic_dev_toolkit.py

⚠️ Notes

This is still a prototype.

Always review diffs before pushing real changes.

Interactive mode may not work in all environments.

The bottom-level argparse CLI runs even if Typer is available.

📄 License

MIT (or update to your preferred license).
