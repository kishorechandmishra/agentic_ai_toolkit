Agentic Dev Toolkit

A minimal, sandbox-safe agentic developer assistant
Intent → Plan → Execution → PR Template

⭐ Overview

The Agentic Dev Toolkit is a lightweight prototype of a real-world agentic developer workflow:

Capture developer intent

Translate it into a structured JSON plan (LLM or local file)

Execute planned steps safely (dry-run by default)

Produce PR-ready output and diffs

It is designed to work in restricted environments, gracefully degrading when dependencies like:

openai

typer

pydantic

gitpython

are missing.

🔥 Highlights
🌐 LLM Optional

Uses OpenAI if installed — otherwise runs entirely offline.

🛡️ Safe Execution

Always dry-run unless --apply-changes is explicitly provided.

🧩 Deterministic Agent Steps

Handles the core step types:

shell

python

git

test

manual

📦 Zero-Dependency Mode

If packages are missing, built-in shims activate automatically.

📝 Auto PR Generator

Every run produces a clean PR body summarizing:

summary

steps

risks

🚀 Quick Start
Mock Plan (Safe for any environment)
python agentic_dev_toolkit.py --mock-plan

Using Developer Intent
python agentic_dev_toolkit.py --intent "Refactor token parser"

Using a JSON Plan File
python agentic_dev_toolkit.py --plan-file ./plan.json

Applying Real Changes (Danger!)
python agentic_dev_toolkit.py --intent "Fix cache invalidation" --apply-changes

🧪 Testing

Run internal unittest suite:

RUN_UNIT_TESTS=1 python agentic_dev_toolkit.py

📐 Architecture Diagram (Mermaid)
flowchart TD

A[User Intent or Plan File] --> B[Plan Generation]
B -->|LLM available| C[OpenAI LLM]
B -->|No LLM| D[Local Plan or Mock Plan]

C --> E[Structured Plan]
D --> E

E --> F[Execution Engine (Dry-Run Default)]
F --> G[Step Handlers]
G --> H[Shell / Python / Git / Test / Manual]

H --> I[Results + Summary]
I --> J[PR Template Generator]
J --> K[Output PR Body + Git Diff]

🧭 Design Principles

Fail safe, not loud → dry-run is default

Deterministic steps → all actions visible before execution

Fully portable → works on machines with zero dependencies

Composable → each step type can be extended independently

Transparent → clear logs + PR templates

🛠 Optional Dependencies

Install for full features:

pip install openai pydantic typer GitPython


Enable LLM planning:

export OPENAI_API_KEY="your_api_key_here"

🧰 Project Structure
agentic_dev_toolkit.py   # Full agent system in one file
README.md                 # Documentation
plan.json (optional)      # Custom plans

📄 License

MIT
