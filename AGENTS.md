# AGENTS.md

This file is for coding agents working in this repository.

## Scope

- Repository root: `/home/vamper/robot skill`
- Main project code lives in `无导航纯RL底层运动器/`
- Other top-level folders such as `EVAL_result/`, `GIF_FOR_GOODS/`, and `成果/` are artifact or reference folders, not the primary source tree
- There is no existing repo-specific Cursor or Copilot instruction set to preserve beyond this file

## Project Summary

- This is a Python reinforcement learning project built around Gymnasium, Pygame, NumPy, Stable-Baselines3, and PPO
- The core environment class is `LTLfGymEnv` in `无导航纯RL底层运动器/mutiple_train.py`
- The environment models 2D continuous motion with obstacle avoidance and ordered `Task A -> Task B -> Task C` completion via a simple local DFA
- The main randomized training entry point is `无导航纯RL底层运动器/generalization_train.py`
- There is also a fine-tuning script in `无导航纯RL底层运动器/finetune_train.py`
- Evaluation and replay tooling lives in `无导航纯RL底层运动器/transfer_eval.py` and `无导航纯RL底层运动器/record_agent.py`
- Training-curve plotting helpers live in `无导航纯RL底层运动器/plot_training_curves.py` and `无导航纯RL底层运动器/plot_comprehensive_curves.py`

## Important Reality Check

- There is no packaged build system, no `pyproject.toml`, no `setup.cfg`, no `pytest.ini`, and no committed linter config
- There is no automated pytest suite in the repository today
- "Testing" in this repo usually means running a script directly or doing lightweight syntax validation
- Avoid inventing folders like `continue/`, `sample/`, or `continue/test/`; they are not part of this checkout

## Environment And Dependencies

- Python 3 is assumed
- Runtime dependencies used by the main project: `gymnasium`, `stable-baselines3`, `numpy`, `pygame`, `imageio`, `torch`, `tensorboard`, `matplotlib`
- `record_agent.py` needs `imageio`
- The plotting scripts need `matplotlib`

## Setup Commands

- Basic install for core training/eval flow:
  - `pip install gymnasium stable-baselines3 numpy pygame imageio torch tensorboard matplotlib`
- If SB3 extras are missing locally, install them explicitly rather than assuming a repo-managed environment

## Build Commands

- There is no formal build step
- Safe validation command after Python edits:
  - `python -m py_compile "无导航纯RL底层运动器"/*.py`
- Treat model training, evaluation, GIF export, and plotting as script execution rather than build steps

## Lint Commands

- No repo-defined lint workflow is committed
- Do not claim that `ruff`, `flake8`, `black`, or `isort` are required unless you add and document them
- Safe lightweight validation remains:
  - `python -m py_compile "无导航纯RL底层运动器"/*.py`
- If the user explicitly wants ad hoc linting and local tools exist, a reasonable optional command is:
  - `ruff check "无导航纯RL底层运动器"`

## Test Commands

- There is no automated test suite
- Practical verification commands are script-based:
  - `python "无导航纯RL底层运动器"/mutiple_train.py`
  - `python "无导航纯RL底层运动器"/generalization_train.py`
  - `python "无导航纯RL底层运动器"/finetune_train.py`
  - `python "无导航纯RL底层运动器"/transfer_eval.py --help`
  - `python "无导航纯RL底层运动器"/record_agent.py --help`
- Because training is expensive, prefer `--help` checks or syntax compilation unless the user explicitly wants a run

## Single-Test Guidance

- There is no `pytest path::test_name` style selector in this repo
- For a quick non-training verification, use:
  - `python -m py_compile "无导航纯RL底层运动器"/*.py`
- For evaluation-path verification, use:
  - `python "无导航纯RL底层运动器"/transfer_eval.py --help`
- For replay-path verification, use:
  - `python "无导航纯RL底层运动器"/record_agent.py --help`

## Key Code Facts

- `LTLfGymEnv` supports `observation_mode="absolute"` and `observation_mode="relative"`
- The single-frame observation dimensions are 23 for `absolute` and 27 for `relative`
- Training scripts stack 4 frames with `VecFrameStack`, so policy inputs are typically 92 or 108 dimensions
- Lidar uses 16 rays with normalized distances
- The environment exposes stagnation and front-sector blockage features in observations and reward shaping
- `generalization_train.py` randomizes layouts from `LAYOUT_LIBRARY`, applies optional flips, and scales perturbations with a curriculum
- `transfer_eval.py` auto-infers `observation_mode` and `n_stack` from the loaded model when possible

## Common Commands

- Base single-layout training:
  - `python "无导航纯RL底层运动器"/mutiple_train.py`
- Randomized generalization training:
  - `python "无导航纯RL底层运动器"/generalization_train.py`
- Fine-tuning from a saved model:
  - `python "无导航纯RL底层运动器"/finetune_train.py`
- Transfer evaluation with default best-model path:
  - `python "无导航纯RL底层运动器"/transfer_eval.py`
- Save evaluation JSON explicitly:
  - `python "无导航纯RL底层运动器"/transfer_eval.py --save-json transfer_eval_results.json`
- Record a GIF replay:
  - `python "无导航纯RL底层运动器"/record_agent.py --scenario ultimate_generalization_test --gif-name agent_trajectory.gif`
- Plot evaluation curves from SB3 `evaluations.npz`:
  - `python "无导航纯RL底层运动器"/plot_training_curves.py`
  - `python "无导航纯RL底层运动器"/plot_comprehensive_curves.py`

## Style Guidelines

- Follow existing repository style before introducing new abstractions
- Prefer small, direct Python scripts over framework-heavy restructuring
- Keep RL environment logic explicit and easy to trace step by step
- Match the current script-first workflow unless the user asks for deeper refactoring

## Imports

- Use standard library imports first, then third-party imports, then local imports
- Existing local imports typically use direct sibling imports such as `from mutiple_train import ...`
- Add new imports only when needed and avoid unused imports in touched files

## Formatting

- Follow PEP 8 style with 4-space indentation
- Keep lines readable rather than aggressively compact
- Use blank lines between helpers, classes, and major logic blocks
- Match surrounding style; some files are prototype-like but new edits should still stay tidy

## Types

- Add type hints where the file already uses them, especially in `无导航纯RL底层运动器/generalization_train.py` and `无导航纯RL底层运动器/finetune_train.py`
- Avoid noisy full-file type retrofits in older prototype sections unless already making broad changes
- Keep helper signatures explicit when that improves readability

## Naming Conventions

- Classes use `PascalCase`
- Functions, methods, and variables use `snake_case`
- Constants use `UPPER_CASE`
- Preserve the existing filename typo `mutiple_train.py`; do not rename it casually because sibling scripts import it directly

## Environment And RL Conventions

- Preserve Gymnasium contracts for `reset`, `step`, `render`, and `close`
- `reset` should return `obs, info`
- `step` should return `obs, reward, terminated, truncated, info`
- Keep observation dimensions synchronized with `observation_mode`
- If you change observation structure, also update `OBSERVATION_DIMS` and model-shape inference in `无导航纯RL底层运动器/transfer_eval.py`
- If you change environment state fields or reward shaping, verify lidar, stagnation detection, front-sector logic, rendering, and DFA progression still agree

## Error Handling

- Prefer explicit `ValueError` for invalid user-provided options, matching `record_agent.py` and `transfer_eval.py`
- Fail early when model observation shape does not match the requested observation configuration
- Keep error messages actionable and include the invalid value plus valid alternatives when possible
- Do not silently swallow exceptions in training or evaluation code unless the file already uses a deliberate recovery path

## File And Path Handling

- Prefer `pathlib.Path` for new file-writing helpers when practical; `transfer_eval.py` already uses it for JSON output
- Keep model paths configurable via CLI flags
- Preserve current artifact locations unless the user asks for a cleanup or reorganization
- Be careful with paths that include Chinese characters; quote them correctly in shell commands

## Agent Editing Guidance

- Make the smallest safe change that matches current patterns
- Do not silently rename public files, artifact directories, or CLI flags
- Preserve Chinese-language user-facing text consistency inside touched files
- If you introduce new tooling, document whether it is required or optional

## Verification Checklist For Agents

- Run `python -m py_compile "无导航纯RL底层运动器"/*.py` after non-trivial Python edits when possible
- If you changed evaluation logic, run `python "无导航纯RL底层运动器"/transfer_eval.py --help`
- If you changed GIF recording logic, run `python "无导航纯RL底层运动器"/record_agent.py --help`
- If you changed plotting scripts, run the relevant script with an existing `evaluations.npz` path or at least confirm the import path is still valid
- If you changed environment dynamics or reward shaping, prefer at least one direct environment or training smoke run if the user expects runtime verification

## What To Avoid

- Do not claim there is a pytest suite when there is not
- Do not assume repo-wide lint tooling exists
- Do not rename `mutiple_train.py` unless the user explicitly asks for a coordinated migration
- Do not refer to nonexistent files such as `continue/plot_transfer_results.py`
- Do not add heavyweight architecture patterns unless they solve a real problem in the current script-oriented workflow
