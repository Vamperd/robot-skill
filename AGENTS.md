# AGENTS.md

This file is for coding agents working in this repository.

## Scope

- Repository root: `/home/vamper/robot skill`
- Main project code lives in `continue/`
- Older exploratory examples live in `sample/`
- There is no existing `AGENTS.md` to preserve
- No Cursor rules were found in `.cursor/rules/` or `.cursorrules`
- No Copilot rules were found in `.github/copilot-instructions.md`

## Project Summary

- This is a Python reinforcement learning project built around Gymnasium, Pygame, NumPy, and Stable-Baselines3
- The primary environment class is `LTLfGymEnv` in `continue/mutiple_train.py`
- The main generalized training entry point is `continue/generalization_train.py`
- Evaluation/reporting scripts are `continue/transfer_eval.py`, `continue/record_agent.py`, and `continue/plot_transfer_results.py`
- Files under `continue/test/` are interactive demo scripts, not a pytest suite

## Important Reality Check

- There is no packaged build system, no `pyproject.toml`, no `setup.cfg`, no `pytest.ini`, and no committed lint config
- There is no formal single-command test runner in the repo today
- "Single test" in this codebase usually means running one Python script directly, especially a file under `continue/test/`
- Avoid inventing repo tooling that is not present

## Environment And Dependencies

- Python 3 is assumed
- Core runtime dependencies mentioned in `continue/README.md`: `gymnasium`, `stable-baselines3`, `numpy`, `pygame`, `imageio`, `torch`, `tensorboard`
- `sample/phase3_single_agent_rl.py` also imports `spot`

## Setup Commands

- Basic install from repo documentation: `pip install gymnasium stable-baselines3 numpy pygame imageio`
- Likely also needed for the main project: `pip install torch tensorboard`
- Likely needed only for `sample/phase3_single_agent_rl.py`: `pip install spot`

## Build Commands

- There is no formal build step
- The nearest equivalents are validation or artifact-generation commands:
  - `python -m py_compile continue/*.py continue/test/*.py sample/*.py`
  - `python continue/plot_transfer_results.py --input continue/transfer_eval_results.json --output continue/transfer_eval_report.html`
- Treat model training as execution, not build

## Lint Commands

- No repo-defined linter configuration is committed
- Do not claim that `ruff`, `flake8`, `black`, or `isort` are required unless you add and document config for them
- Safe lightweight validation command: `python -m py_compile continue/*.py continue/test/*.py sample/*.py`
- If the user explicitly wants linting and local tools are available, a reasonable ad hoc command is `ruff check continue sample`
- If you use ad hoc linting, say clearly that it is not repository-defined

## Test Commands

- There is no automated pytest-style test suite in the repository
- Existing tests are runnable scripts, mostly under `continue/test/`
- Run current demo tests individually:
  - `python continue/test/base_env.py`
  - `python continue/test/reward_base_env.py`
  - `python continue/test/muti_reward_base_env.py`
- Training smoke path: `python continue/mutiple_train.py`
- Generalized training: `python continue/generalization_train.py`
- Transfer evaluation: `python continue/transfer_eval.py --model-path continue/generalization_eval_best/best_model.zip`
- GIF replay capture: `python continue/record_agent.py --model-path continue/generalization_eval_best_v3/best_model.zip --scenario hard_ood_layout --gif-name agent_trajectory.gif`

## Single-Test Guidance

- For a single visual environment test: `python continue/test/base_env.py`
- For a single reward-shaping test: `python continue/test/reward_base_env.py`
- For a single multi-task reward test: `python continue/test/muti_reward_base_env.py`
- There is no test selector like `pytest path::test_name`
- If you add pytest later, document exact single-test commands here

## Evaluation And Reporting Commands

- Save JSON: `python continue/transfer_eval.py --model-path continue/generalization_eval_best/best_model.zip --save-json continue/transfer_eval_results.json`
- Auto-detect observation mode: `python continue/transfer_eval.py --model-path continue/generalization_eval_best/best_model.zip`
- Force relative mode: `python continue/transfer_eval.py --model-path continue/generalization_eval_best_v3/best_model.zip --observation-mode relative`
- Render first scenario: `python continue/transfer_eval.py --model-path continue/generalization_eval_best_v3/best_model.zip --render-first`
- Build HTML: `python continue/plot_transfer_results.py --input continue/transfer_eval_results.json --output continue/transfer_eval_report.html`

## Style Guidelines

- Follow existing repository style before imposing new abstractions
- Prefer small, direct Python scripts over framework-heavy structure
- Keep RL environment logic explicit and easy to trace step-by-step
- Preserve the current script-first workflow unless the user asks for a larger refactor

## Imports

- Use standard library imports first, then third-party imports, then local imports
- Typical local import pattern is `from mutiple_train import ...` when working inside `continue/`
- Use `from typing import ...` only when type hints are already being used in that file
- Avoid unused imports; existing files have a few inconsistencies, but new changes should be cleaner

## Formatting

- Follow PEP 8 style with 4-space indentation
- Keep lines readable rather than aggressively compact
- Use blank lines to separate helper functions, classes, and major logical blocks
- Match the surrounding file style; some files are more prototype-like, others are more structured
- Do not introduce formatter-specific rewrites unless the user asks for them

## Types

- Add type hints where the file already uses them, especially in `continue/generalization_train.py` and `continue/transfer_eval.py`
- In older prototype files, avoid noisy full-file type retrofits unless you are already touching many lines there
- Keep public helper return types explicit when inference is not obvious

## Naming Conventions

- Classes use `PascalCase`
- Functions, methods, and variables use `snake_case`
- Constants use `UPPER_CASE`
- Scenario names and task labels are string literals such as `"Task A"` and `"hard_ood_layout"`; preserve existing naming unless there is a migration plan
- Be careful with the existing filename typo `mutiple_train.py`; do not rename it casually because multiple scripts import it directly

## Environment And RL Conventions

- Preserve Gymnasium method contracts for `reset`, `step`, `render`, and `close`
- Return `obs, info` from `reset`
- Return `obs, reward, terminated, truncated, info` from `step`
- Keep observation dimensions synchronized with `observation_mode`
- If you change observation structure, also update evaluation mode checks in `continue/transfer_eval.py`
- If you change environment state fields, verify render, lidar, reward shaping, and DFA progression still agree

## Error Handling

- Prefer explicit `ValueError` for invalid user-provided options, matching `continue/transfer_eval.py` and `continue/record_agent.py`
- Fail early when a model's observation shape does not match the requested observation mode
- Use clear, actionable error messages that mention the bad value and the valid alternatives
- Avoid swallowing exceptions in training or evaluation code unless there is a strong recovery path

## File And Path Handling

- Prefer `pathlib.Path` for file IO in utility scripts, as already done in `continue/transfer_eval.py` and `continue/plot_transfer_results.py`
- Keep model-path arguments configurable via CLI flags
- When adding outputs, use predictable filenames in the working directory unless the repo already has a dedicated output folder

## Agent Editing Guidance

- Make the smallest safe change that matches current patterns
- Do not silently rename public files, model artifacts, or CLI flags
- Be cautious with Chinese-language user-facing text; preserve language consistency within the touched file
- If you introduce new tooling, document whether it is required or optional

## Verification Checklist For Agents

- Run `python -m py_compile continue/*.py continue/test/*.py sample/*.py` after non-trivial Python edits when possible
- If you changed evaluation logic, run `python continue/transfer_eval.py --help`
- If you changed report generation, run `python continue/plot_transfer_results.py --help`
- If you changed GIF capture CLI, run `python continue/record_agent.py --help`
- If you changed environment dynamics, at minimum run one relevant script under `continue/test/`

## What To Avoid

- Do not claim there is a pytest suite when there is not
- Do not assume repo-wide lint tooling exists
- Do not rename `mutiple_train.py` unless the user explicitly asks for the migration and all imports are updated
- Do not add heavyweight architecture patterns unless they solve a real pain point in the current scripts
