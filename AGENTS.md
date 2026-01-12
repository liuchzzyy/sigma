# Agentic Coding Guidelines for SIGMA

This file provides instructions for AI agents (and human developers) working on the **SIGMA** repository. Follow these guidelines to ensure consistency, reliability, and modern best practices.

## 1. Environment & Tooling

We use modern Python tooling. DO NOT use legacy tools like `pip` directly or `setup.py`.

- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Build Backend**: `hatchling`
- **Python Version**: `>=3.11`
- **Linting**: `ruff`
- **Formatting**: `ruff format`
- **Type Checking**: `pyright` (via `uv run pyright` or VS Code extension)

### Common Commands

| Task | Command | Notes |
| :--- | :--- | :--- |
| **Install** | `uv sync` | Installs dependencies from `pyproject.toml` |
| **Lint** | `uv run ruff check .` | Checks code quality |
| **Lint (Fix)** | `uv run ruff check . --fix` | Auto-fixes simple linting issues |
| **Format** | `uv run ruff format .` | Formats code to standard style |
| **Type Check** | `uv run pyright` | Strict type checking |
| **Test** | `uv run pytest` | Runs all tests |
| **Single Test** | `uv run pytest tests/test_smoke.py` | Run specific test file |

## 2. Project Structure

The project is organized into modular components. **DO NOT** create files in the root or generic `src` folders.

- `sigma/core/`: Core data structures and training loops (e.g., `Experiment`, `FeatureDataset`).
- `sigma/processing/`: Main scientific logic (e.g., `segmentation.py`).
- `sigma/models/`: Neural network architectures (Autoencoders, VAEs).
- `sigma/utils/`: shared utilities (`io`, `visualization`, `physics`, `signal`).
- `sigma/gui/`: Interactive Jupyter widgets and plotting tools.
- `tests/`: Unit tests (pytest style) [Currently Missing - Future Implementation].

## 3. Code Style & Conventions

### Imports
- **Grouping**: Standard library -> Third-party -> Local application (enforced by `ruff`).
- **Style**: Absolute imports preferred.
  ```python
  # Good
  from sigma.core.data import FeatureDataset
  
  # Bad
  from ..core.data import FeatureDataset
  ```

### Formatting
- **Line Length**: 88 characters (Black/Ruff standard).
- **Quotes**: Double quotes `"` for strings.
- **Indentation**: 4 spaces.

### Naming
- **Classes**: `PascalCase` (e.g., `PixelSegmenter`, `FeatureDataset`).
- **Functions/Variables**: `snake_case` (e.g., `fft_denoise2d`, `run_model`).
- **Constants**: `UPPER_CASE` (e.g., `K_FACTORS_120KV`).
- **Private**: Prefix with `_` (e.g., `_color_palette`).

### Typing
- **Strictness**: All new code MUST be typed.
- **Style**: Use modern Python 3.10+ syntax (e.g., `list[str]`, `str | None`) where possible, but `typing` module is acceptable for compatibility.
- **Arrays**: Use `numpy.typing.NDArray` or `np.ndarray` for clarity.

```python
def process_image(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    ...
```

### Error Handling
- Use specific exceptions (e.g., `ValueError`, `TypeError`) instead of generic `Exception`.
- Provide meaningful error messages.
- Use `try/except` blocks narrowly around the code that might fail.

## 4. Testing

> **Note**: The testing infrastructure is currently being set up. The following guidelines are for future reference.

- (Optional) New features can include tests in `tests/` if the directory exists.
- Use `pytest` fixtures for common setup (e.g., creating dummy datasets).
- Mock external dependencies (like heavy `hyperspy` loads) if they are slow or require large files.
- **Warning Suppression**: Upstream deprecation warnings from `hyperspy` and `rsciio` are suppressed in `pyproject.toml`. Do not remove these filters unless the libraries are updated to fix them.

## 5. Deprecation & Compatibility

- When moving or renaming functions, provide a backward-compatibility shim if possible, with a `DeprecationWarning`.
- **Example**:
  ```python
  import warnings
  def old_function():
      warnings.warn("Use new_function instead", DeprecationWarning, stacklevel=2)
      return new_function()
  ```

## 6. Specific Library Notes

- **Hyperspy**: Used for handling spectral data. Ensure versions `>=2.3.0`.
- **Torch**: Use device-agnostic code (`device = "cuda" if torch.cuda.is_available() else "cpu"`).
- **Numpy**: Avoid `np.matrix`; use `np.array`. Be aware of version `<2.0.0` constraint for Numba compatibility (though modern Numba may support 2.0).
- **Numba**: Explicitly required (`>=0.9.0` per pyproject.toml) for acceleration in some modules.

## 7. Workflow for Agents

1.  **Explore**: Understanding the task and relevant files.
2.  **Plan**: Break down the task.
3.  **Implement**: Write code (tests optional).
4.  **Verify**:
    -   Run `uv run ruff check .`
    -   Run `uv run pyright`
    -   (Optional) Run `uv run pytest` if tests exist
5.  **Refactor**: Cleanup if verification fails.
