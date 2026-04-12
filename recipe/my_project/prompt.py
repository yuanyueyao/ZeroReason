prompt_A = """## Task: Create a New Python Code Snippet with One Matching Input

You prepare a challenging task for another model (model B). Using standard Python only, design a **new and unique** code snippet where a test subject must use **deep algorithmic reasoning** to predict the **deterministic output** from the given input (like an I.Q. test).

Your submission must include:
1) A **first** code block: **only** imports (optional), optional classes/helpers, and **one** function named exactly **`f`**.
2) A **second** block: **only** the arguments that will be passed to `f` — **not** the computed output.

### Code block (```python) — STRICT CONTENT RULES
- The **only** top-level callable entry point must be **`def f(...):`**. Do **not** use another name (wrong: `def find_maximum_sum`, `def solve`, etc.).
- Allowed at top level, in order: `import ...` (stdlib only), optional `class` / helper `def` **only if required**, then **`def f(...):`** whose body ends with **`return ...`**.
- **FORBIDDEN inside the first block** (very common mistakes — do not do these):
  - Any **test harness**: assigning `result = f(...)`, calling `f(...)` at module level, `if __name__ == "__main__":`, etc.
  - **`print(...)`**, **`input(...)`**, logging, or **comments like `# Input`** followed by fake test code.
  - **Parsing strings** that duplicate the job of `f` outside `f` (e.g. `input_args = '1,2,3'` then `f([int(x) for ...])` at top level). Put logic **inside** `f` instead; the second block supplies **ready Python values** as arguments.
  - Anything **after** the final line of `def f` (no trailing top-level statements).
- Nested `def` / classes **inside** `f` are allowed.
- `f` must **return** a value, take **at least one** parameter, be **deterministic**.
- Require **multi-step reasoning** (e.g. trees, heaps, graphs, DP, recursion, backtracking, etc.).
- **AVOID:** randomness, date/time, I/O, printing, external mutable state. ~10s CPU.

### Second block (```input) — STRICT
- Must contain **only** a comma-separated list of **Python expressions** that are the **arguments to `f`**, in order — what you would write inside `f(` `)`.
- **WRONG:** the expected numeric/text **answer** (e.g. `13`), or a bare output label. **RIGHT:** values such as `[-1, 2, -3, 4]` or `'John', {'age': 20}`.

### Formatting — exactly two fences, in order, nothing else
1) ```python — **only** the rules above (imports + optional types + `def f...`).
2) ```input — **only** the argument list for one call `f(...)`.

### Good example
```python
def f(name: str, info: dict) -> str:
    # logic only inside f
    return str(len(name)) + str(info.get("age", 0))
```
```input
'John', {'age': 20, 'city': 'New York'}
```

### Bad example (do NOT output anything like this)
- First block ends with `print(result)` or `result = find_maximum_sum(...)` — **invalid**.
- Second block is `13` — that is an **answer**, not **inputs** — **invalid**.

### Quality
- Executable as: load first block, then `f(<args from second block>)`.
- Prefer non-trivial algorithmic depth; avoid trivial one-liners.

Briefly plan internally, then output **only** the two fenced blocks (no other text)."""