# AGENTS.md

## Build & Run

```bash
# Install dependencies
poetry install

# Run all tests
poetry run pytest -v

# Run a single test by name
poetry run pytest tests/test_integration.py::test_tick_updates_state -v

# Run tests matching a keyword
poetry run pytest -k "seeding" -v

# Lint (includes isort via ruff's I rules)
poetry run ruff check src/ tests/

# Lint with auto-fix
poetry run ruff check --fix src/ tests/

# Format
poetry run ruff format src/ tests/

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

## Pre-commit Hooks

Three local hooks run on every commit (`.pre-commit-config.yaml`):
1. **ruff lint** -- linting + isort import sorting, with `--fix`
2. **ruff format** -- code formatting
3. **pytest** -- full test suite

All hooks use `poetry run` to invoke locally-installed tools (no remote repos).

## Project Structure

```
src/joust/           # Package source (src-layout)
├── dto.py           # Core data: Team, Match
├── predictor.py     # MatchPrediction, MatchPredictor (ABC), RandomMatchPredictor
├── seeding.py       # Standalone utilities: difficulty_score(), recalculate_seeds()
├── tournament.py    # MatchupStrategy (ABC), PureSeedMatchup, Tournament
├── __init__.py      # Empty
└── __main__.py      # Empty (no CLI yet)
tests/
└── test_integration.py
plans/
└── backlog.md       # Upcoming work (steps 6-10)
```

- **Python >= 3.11**, managed with **Poetry**.
- Runtime dependency: **numpy >= 2.4.3**.
- Dev dependencies: **pytest**, **ruff**, **pre-commit**.

## Domain Context

This is a tournament simulator for CS2 Major-style swiss system tournaments.
The primary reference is the Valve Major Supplemental Rulebook
(`~/personal/counter-strike_rules_and_regs/major-supplemental-rulebook.md`).

Key concepts:
- A **Tournament** represents one stage (e.g., Stage 1 swiss, Stage 2 swiss, Playoffs).
- Separate Tournament instances are created per stage; an external orchestrator passes teams between them.
- Swiss stages: 16 teams, 3 wins to advance, 3 losses to eliminate, up to 5 rounds.
- **Seeding** is recalculated after each round: W-L record > Buchholz difficulty > initial seed.
- **Buchholz/Difficulty Score**: sum of (wins - losses) of all opponents a team has faced.
- Teams are **mutated in-place** (wins, losses, seed). To branch simulations, `deepcopy` the Tournament before calling `tick()`.

## Code Style

### General

- Keep files short. Prefer many small modules over large files.
- Minimal abstraction -- this is a domain-specific simulator, not a framework.
- Use `@dataclass` for data containers. Add computed properties where natural.
- Use ABC + `@abstractmethod` for extension points (strategies, predictors). Do not add `__init__` to ABCs unless state is needed.

### Types & Annotations

- Use Python 3.10+ union syntax: `bool | None`, not `Optional[bool]` or `Union[bool, None]`.
- Use `list[X]`, `dict[K, V]` (lowercase builtins), not `List`, `Dict` from typing.
- Annotate return types on all public methods and properties.
- Dataclass fields use type annotations; provide defaults where sensible (`wins: int = 0`).

### Imports

- Standard library first, then third-party, then `joust.*` -- separated by blank lines.
- Ruff's isort (`I` rules) enforces this automatically. `known-first-party = ["joust"]` is configured.
- Import specific names: `from joust.dto import Team, Match`, not `import joust.dto`.
- Keep imports minimal; only import what's used (ruff `F401` catches unused imports).

### Naming

- Classes: `PascalCase` (`PureSeedMatchup`, `MatchPrediction`).
- Functions and methods: `snake_case` (`difficulty_score`, `predict_winner`).
- Module filenames: `snake_case` (`dto.py`, `tournament.py`).
- Private attributes: single underscore prefix only when truly internal. Prefer public attributes on Tournament (e.g., `match_history`).
- Test functions: `test_<what_it_tests>` with no class wrappers.
- Avoid ambiguous single-letter variable names (ruff `E741`).

### Formatting

- Line length: **100** (configured in `pyproject.toml`).
- Ruff format handles all whitespace, quoting, and trailing commas. Do not fight the formatter.
- Two blank lines between top-level definitions (classes, functions).
- One blank line between methods inside a class.
- Docstrings: single-line where sufficient (`"""Match with no possibility of tie."""`). No docstrings required on obvious methods.

### Error Handling

- Use `assert` with descriptive messages for preconditions (e.g., validating team count).
- Do not use exceptions for control flow.

### Tests

- All tests live in `tests/`. File naming: `test_<module_or_feature>.py`.
- Use plain functions, not test classes.
- Group related tests with `# --- section name ---` comments.
- Helper functions prefixed with underscore: `_make_teams(n=16)`.
- Test names should read as assertions: `test_tick_updates_state`, `test_seeds_update_after_tick`.
- No fixtures or conftest yet -- construct objects directly in each test.

## Architecture Decisions

1. **Mutate in-place**: `Tournament.tick()` updates team records, match outcomes, and seeds directly. Use `copy.deepcopy(tournament)` before `tick()` to branch (e.g., Monte Carlo).

2. **Strategy pattern for matchups**: `MatchupStrategy` subclasses define pairing logic. The strategy receives all teams and full match history -- it is responsible for grouping by W-L pool, avoiding rematches, etc.

3. **Predictor pattern for outcomes**: `MatchPredictor` subclasses decide match winners. Currently only `RandomMatchPredictor` (50/50 coin flip).

4. **Standalone seeding utilities**: `difficulty_score()` and `recalculate_seeds()` are free functions in `seeding.py`, not methods on Tournament. They take teams + history as args.

5. **Team status is derived**: No explicit status enum. Active = `wins < 3 and losses < 3`. Advanced = `wins >= 3`. Eliminated = `losses >= 3`. Filtered via properties on Tournament.

6. **One strategy per round**: `Tournament.round_strategies` is a list indexed by `current_round`. Different pairing logic per round.

7. **External orchestration between stages**: No "Major" class. A script creates each stage's Tournament, runs it, extracts advancers, reseeds, and creates the next stage.

## Backlog

See `plans/backlog.md` for upcoming work: `Tournament.run()`, `SwissMatchupStrategy`, `SingleEliminationStrategy`, and the external orchestrator script.
