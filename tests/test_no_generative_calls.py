"""The central design constraint, enforced against the source.

The brief draws one line carefully: the model must not make the judgement call while the app
relays the verdict. That is easy to honour on day one and easy to lose later, so it is a test
rather than a promise.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "palimpsest"

#: Everything that would mean "we asked a model for text or an opinion".
FORBIDDEN_ATTRS = {"generate", "chat", "complete", "completions", "create_completion"}
FORBIDDEN_NAMES = {"openai", "anthropic", "cohere", "litellm", "ollama", "google.generativeai"}


def python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_at_least_one_source_file_found():
    assert python_files(), "no source files found; the guard below would vacuously pass"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_no_generation_call(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in FORBIDDEN_ATTRS, (
                f"{path.name}:{node.lineno} calls .{node.func.attr}() -- the scoring path "
                "must never ask a model to produce text or a verdict"
            )


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_no_llm_api_clients_imported(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            assert root not in FORBIDDEN_NAMES, f"{path.name} imports {name}"


def test_scorer_uses_a_single_forward_pass():
    """The observer is called as a function, not driven as a generator."""
    source = (SRC / "scorer" / "local_lm.py").read_text(encoding="utf-8")
    assert "torch.no_grad()" in source
    assert ".generate(" not in source
    assert "apply_chat_template" not in source
