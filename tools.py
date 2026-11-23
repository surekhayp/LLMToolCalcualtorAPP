#import tools from langchain
from langchain_core.tools import tool
# ode to define tools for llm for perforing add, subtract , divide and multiply operations

# 1. Define calculator functions using the @tool decorator
@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract the second number (b) from the first number (a)."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def divide(a: float, b: float) -> float | str:
    """Divide the first number (a) by the second number (b)."""
    if b == 0:
        return "Error: Division by zero is not allowed."
    return a / b
