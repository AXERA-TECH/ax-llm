#!/usr/bin/env python3
import argparse
import ast
import operator
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mcp.server.fastmcp import FastMCP


WORKSPACE = Path(os.environ.get("MCP_DEMO_WORKSPACE", os.getcwd())).resolve()
mcp = FastMCP("axllm-demo-tools")


def _safe_eval(node):
    binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in binary_ops:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("power exponent is too large")
        return binary_ops[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in unary_ops:
        return unary_ops[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only numeric arithmetic expressions are allowed")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression, such as '(12 + 8) * 3'."""
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree)
    return f"{expression} = {result}"


@mcp.tool()
def get_time(timezone: str = "Asia/Shanghai") -> str:
    """Return the current time for an IANA timezone, such as Asia/Shanghai or UTC."""
    now = datetime.now(ZoneInfo(timezone))
    return now.isoformat(timespec="seconds")


@mcp.tool()
def read_text_file(path: str, max_chars: int = 2000) -> str:
    """Read a UTF-8 text file under the configured demo workspace."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    requested = (WORKSPACE / path).resolve()
    if not requested.is_relative_to(WORKSPACE):
        raise ValueError(f"path must stay under workspace: {WORKSPACE}")
    if not requested.is_file():
        raise FileNotFoundError(str(requested))

    text = requested.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[truncated, total_chars={len(text)}]"
    return text


def main():
    parser = argparse.ArgumentParser(description="Small MCP stdio server for axllm OpenAI API demos.")
    parser.add_argument("--workspace", default=os.getcwd(), help="Root directory allowed for read_text_file.")
    args = parser.parse_args()

    global WORKSPACE
    WORKSPACE = Path(args.workspace).resolve()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
