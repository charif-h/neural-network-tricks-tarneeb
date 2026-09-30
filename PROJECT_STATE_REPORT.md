# Project State Report

**Repository:** neural-network-tricks-tarneeb
**Date:** 2026-09-24
**Scope:** code quality, documentation clarity, and other project-health factors
**Method:** full read of every source and doc file, plus actually executing the entry points in the project `.venv` (Python 3.11, TF/Keras 2.15, NumPy 1.26).

---

## 1. Executive Summary

The project is **a well-documented skeleton that does not currently run**. Every entry point (`Game.py`, `example.py`, `Tarneeb/GTarneeb.py`) fails, and the previous "evaluation" passes (`CODE_EVALUATION.md`, `EVALUATION_SUMMARY.md`, `VERIFICATION_REPORT.md`) reported success on the basis of *syntax checks only*, which is why they missed this. The most recent documentation/docstring work appears to have been applied by a mechanical process that **swapped or scrambled method bodies** in two core files (`TarneebPlayer.py`, `Turn.py`), leaving them syntactically valid but logically broken.

| Area | Grade | One-line verdict |
|---|---|---|
| Runnability | **F** | All three entry points crash |
| Code correctness | **D-** | Method bodies misplaced; missing methods; dead code; side-effecting bugs |
| Code style / structure | **C-** | Inconsistent naming, module-level scripts, globals, no tests |
| Docstring coverage | **B** | Good coverage, but several docstrings describe the wrong code |
| README / user docs | **C** | Nice game and NN-input explanation; instructions don't work; inaccurate claims |
| Repo hygiene | **C-** | Nested duplicate clone, tracked `.pyc`, redundant report files |
| Testing / CI | **F** | None |

---

## 2. Verified Runtime Failures

I ran each entry point (with `PYTHONIOENCODING=utf8` to rule out Windows console noise):

| Command | Result |
|---|---|
| `python Game.py` | `AttributeError: 'Player' object has no attribute 'win'` (also calls the non-existent `getScoreTarneeb`) |
| `python example.py` | Fails with `UnicodeEncodeError` on Windows default console (suits `♣♦♠♥`); the `try/except` swallows it and prints a misleading "install dependencies" hint. Note: with UTF-8 forced it does run, but see 3.5 |
| `python Tarneeb/GTarneeb.py` (as the README says) | `ModuleNotFoundError: No module named 'Cards'` (script dir, not repo root, is on `sys.path`) |
| `PYTHONPATH=. python Tarneeb/GTarneeb.py` | `NameError: name 'scores' is not defined` in `TarneebPlayer.getResult` |
| `TarneebPlayer.bid()` called directly | Same `NameError` |

So the README's "Usage" section is not currently reproducible under any configuration.

---

## 3. Code Quality Findings

### 3.1 Critical: scrambled method bodies (`Tarneeb/TarneebPlayer.py`)

The bodies of methods appear to have been shuffled relative to their names/docstrings:

- `bid()` (docstring: "Make a bidding decision... Returns int") contains the **colored round-result print** logic and returns nothing.
- `getResult()` (docstring: "Calculate the score result for the round") contains the **bidding neural-network inference**, referencing `scores`, `biddings`, `tarneeb` which are not in its scope → `NameError`.
- `print_player_round_results()` (docstring: "Display the player's round results with color coding") contains the **score calculation** (doubling/negation) and prints nothing.
- `roundOver()` calls `getResult()`, which is the broken one above, so scoring is broken too.

Net effect: the bid → play → score → train cycle cannot execute.

### 3.2 Critical: scrambled method bodies (`Tarneeb/Turn.py`)

- `winner()` (docstring: determines the winning card) actually contains the **string-formatting code of `__str__`** and returns a string; it references `self.winCard` before it is ever set.
- `__str__` contains a `return`, then **unreachable code** below it, which is the real winner-determination logic. So `winCard`, `winnerId`, `winCardId` are never set; `__str__` itself would raise `AttributeError`.
- `__repr__` contains the body of what `GTarneeb.py` calls **`turn_to_matrices(players)`**, which does not exist as a method. That body also references a global `players` that does not exist in this module.
- `Turn.__init__` calls `self.winner()` and `self.playing_loss_function()`; the latter reads `self.winCard`, which is never set → crash.

### 3.3 Other correctness bugs

| Location | Problem |
|---|---|
| `Player.py` | `Player.__init__` does not initialise `hand`, so `repr()`/`playCard` before `setHand` raises `AttributeError`. No `win()` / `getScoreTarneeb()` although `Game.py` uses them. `chooseCard` prints `'do not ...'` (leftover debug). Mutable default argument `cards=[]`. |
| `Cards/StandarDeck.py: winner()` | Appends a *list* (`playedCards`) into `self.cards`, corrupting the deck with a non-Card element (flagged as a TODO). Ignores the `tarneeb` parameter it documents. |
| `Cards/Card.py` | `largerThan` semantics were changed ("Fixed: was returning True") without tests; `StandarDeck.winner` relies on the old asymmetry, so trick-winner logic in the base game needs re-verification. `__eq__` crashes on non-Card comparison; `__lt__` sorts by the suit *symbol string*, which is a fragile ordering. `__hash__` hashes strings (slow, unnecessary). `valueChar()` derives the letter by string-parsing `str(enum)` (breaks on Python 3.11+ enum `str` changes in some cases). |
| `Cards/Card.py` | `CardType.__new__(cls, value, name)` swaps the conventional meaning of `value`/`name`; `CardType.value` is the *symbol* and `.id` is the number. Confusing and the source of the `.type.value` idioms scattered around. |
| `TarneebPlayer.playCard` | Prints the full input matrix on every card play; the neural network is not used at all (random choice). Uses `self.number_of_won_turns / self.bidding` (guarded), but mutable default args again (`scores=[0,0,0,0]` ...). |
| `TarneebPlayer.play`, `playModelInput` | Unfinished stubs (`np.concatenate((tarneeb))`, unused variables) that would crash if called. |
| `TarneebPlayer.__init__` | `self.history = np.zeros((52, 68))` allocated but never used. Builds a Keras model per player at construction (slow start). |
| `GenModel.py` | Docstring says "a compiled model" (true) but the first layer is hard-coded to 64 units regardless of `inputs`; `layers`, variable `linputs` naming misleading (units, not inputs). `actiFunction.SOFTMAX` / `EXPONENTIAL` in the random pool for a hidden/output layer of a single-output regression is an unsafe choice (softmax over 1 unit is constant 1). `normalLawCat` shadows built-in `list`. `hard_sigmoid` is removed in newer Keras. `normalActivation` may return an `Enum` (default) or a `str` depending on path. File lacks trailing newline. |
| `GTarneeb.py` | **Whole game runs at import time** (no `if __name__ == "__main__":`), heavy use of module-level globals (`players`, `standardeck`, `cards_record`, `round_loss`). `distripute_and_bid` reads the global `standardeck` created in the main loop. `clearHands` builds a new deck and tarneeb and throws them away (dead code) and sets `p.prediction` (never used). `playRound` calls `turn.turn_to_matrices` twice per turn (once discarded) and calculates an unused `input_matrix`. `loss_t` is unused and compares numpy arrays with `==` (ambiguous truth value). `exit()` is left in the training loop after the first round, so the loop can never run more than one round or one game. Bid-sum check of `< 11` is a hard-coded literal though `constants.MIN_BID_SUM` exists. |
| Tarneeb selection | The rules in README say the tarneeb is the *opposite suit of the same colour* of the last card; code uses `standardeck.cards[51].type` directly. Rule/implementation mismatch. |
| Bidding rules | README lists a bidding phase, but there is no real bidding auction: bids are independent NN outputs, no "highest bidder chooses trump", no team scoring (scores are per *player*, whereas Tarneeb is played in teams of 2). |
| Training design | Playing model is trained on random targets then on the *cards played* as labels (imitating random play); the per-turn `loss` dict is computed but never used as a loss. Bidding target `(tricks-2)/11` can be negative/`>1` and the output activation can be random (see `GenModel`). No RL loop exists yet despite the stated objective. |

### 3.4 Style and structure

- **Naming:** mix of `camelCase` (`setHand`, `handToArray`), `snake_case` (`card_to_matrix`), and PascalCase functions (`Model`, `Turn()`); typos `StandarDeck`, `distripute`, `normaLawInt`. Documented as known and kept "for backward compatibility" in a project with no external users.
- **Imports:** `import Player` / `import GenModel` are top-level, only work when run from repo root; `Tarneeb/GTarneeb.py` needs `PYTHONPATH`. No `pyproject.toml`/package install.
- **Constants:** `constants.py` exists but is **imported nowhere**; magic numbers (`13`, `41`, `52`, `68`, `64`) are still used everywhere. Its comments about `PLAYING_INPUT_DIM = 68` (4+52+12) conflict with the README's list (tarneeb one-hot, scores, bids, wins, ...).
- **Unused imports/vars:** `logging` INFO spam plus `print` mixing; `random`, `math`, `os`, `np` unused in places; `Xbid`/`Ybid` initialised twice.
- **Debug output** in library code (`print('turn analysis', ...)`, `print('player input matrix', ...)`) floods the console.
- **No type hints**, no linter/formatter config (the "development dependencies" in `requirements.txt` are commented out).
- **Docstrings vs code drift:** because bodies were swapped (3.1, 3.2) several docstrings are now *false*; a wrong docstring is worse than none.

### 3.5 Portability

- Suit symbols (`♣♦♠♥`) crash `print` on Windows consoles using cp1252 (reproduced). Needs `sys.stdout.reconfigure(encoding="utf-8")`, or an ASCII fallback.
- `example.py` catches every `Exception` and blames dependencies, hiding the real error.
- `Tarneeb/__pycache__/TarneebPlayer.cpython-37.pyc` is **tracked in git** despite `.gitignore` covering `__pycache__/`.

---

## 4. Documentation Assessment

### Strengths
- README clearly explains Tarneeb-41 rules and the NN input design; a good on-ramp for a newcomer to the game.
- Nearly every module/class/function has a docstring with Args/Returns.
- `CONTRIBUTING.md` exists; `constants.py` is a good idea.
- Honest "Future Work" list.

### Problems
1. **Instructions that don't work.** `python Game.py` and `python Tarneeb/GTarneeb.py` both fail (§2). No troubleshooting or note on running from repo root / `PYTHONPATH`.
2. **Inaccurate claims.** README says the project "uses reinforcement learning" and describes the playing network input as 7 items (tarneeb one-hot, scores, bids, wins, ...) - the implemented input is 68 floats with a different layout. The playing network is not actually used. The bidding phase and team scoring described are not implemented. `example.py` says training will "Train bidding models... Track game statistics" (it doesn't run any of that).
3. **Python version mismatch.** README says 3.8+; the `.venv` is 3.11 with Keras 2.15; `requirements.txt` pins `keras<3` and `tensorflow<3` (mutually unpinned, could resolve to inconsistent pairs), and `hard_sigmoid`/`input_dim` behaviours differ across versions. Untested on 3.8.
4. **Redundant, self-congratulatory reports.** `CODE_EVALUATION.md`, `EVALUATION_SUMMARY.md`, `VERIFICATION_REPORT.md` overlap heavily; the last declares "✅ PASSED" based on syntax only, which is misleading. `CODE_EVALUATION.md` lists issues (naming, `largerThan` bug) mostly *documented rather than fixed*. They clutter the repo root and go stale immediately.
5. **README typos and grammar:** "interrested", "surroundend", "form the player", "Partners set facing", "as follow". Rules text is also imprecise (dealing order, trump selection).
6. **Missing:** LICENSE file (README says "open source" without a licence, which legally means all rights reserved), architecture/data-flow diagram, how to save/load models, expected output, glossary of camelCase API, changelog.
7. **Docstring notes as TODOs** ("seems incorrect and should be reviewed") document bugs rather than fixing them.
8. `CONTRIBUTING.md` promises tests/linting workflows that don't exist in the repo.

---

## 5. Other Important Observations

- **Repo hygiene:** an untracked, full **nested clone** `neural-network-tricks-tarneeb/` sits inside the repo (it has its own `.git`). It's byte-identical to the parent (except `.git`) and will confuse tooling, imports, and editors. Remove or `.gitignore` it.
- **`.venv` is inside the project folder** (ignored, fine), but it contains Python 3.11, whereas a tracked `.pyc` is from 3.7: environments are inconsistent.
- **No tests, no CI.** The single most valuable addition; the card/trick logic is small and highly testable (see recommendations).
- **No reproducibility:** no random seeds; NN architecture is randomised per player (`GenModel`), so runs cannot be compared.
- **No model persistence:** trained models are never saved (though README lists it as future work).
- **Performance:** `predict()` per call in a Python loop, one `fit` per player per round with batch of 4 samples; will be very slow for real training. Consider vectorised batches and `model(x, training=False)`.
- **Security/safety:** no external input or network code; low risk. `exit()` in library-style code is the only questionable control flow.
- **Git history:** last commits are "Add verification report / evaluation summary / clean up" i.e. meta-work about quality rather than functional progress.

---

## 6. Recommendations (prioritised)

### P0 - make it run (a few hours)
1. **Restore the swapped method bodies** in `TarneebPlayer.py` (`bid` ↔ `getResult` ↔ `print_player_round_results`) and `Turn.py` (`winner`, `__str__`, `__repr__`→`turn_to_matrices(players)`). Use git history (`git log -p Tarneeb/`) to find the last working version.
2. Wrap `GTarneeb.py` in `main()` with `if __name__ == "__main__":`; remove the stray `exit()`; pass `standardeck` explicitly.
3. Fix `Game.py`/`example.py`: add `Player.win`/scoring or delete `Game.py` in favour of `example.py`; do not swallow all exceptions.
4. Make imports work from anywhere: add `pyproject.toml` and `pip install -e .`, or run via `python -m Tarneeb.GTarneeb` and document it.
5. Force UTF-8 output (or ASCII fallback) for suit symbols.
6. Untrack `*.pyc`; delete the nested clone.

### P1 - safety net
7. Add `pytest` tests: deck has 52 unique cards; `distribute` sizes; `winner()` with/without trump and follow-suit; `cardId` bijection 0-51; scoring rules (bid met, ≥7 doubling, negative); `Turn` winner. A smoke test that plays one random round end-to-end.
8. Add GitHub Actions (pytest + ruff) on 3.10/3.11.
9. Add `ruff` + `black` config; enable type hints incrementally (`mypy` on `Cards/` first).

### P2 - clarity
10. Rename to PEP 8 (`StandardDeck`, `distribute`, `snake_case` methods) - safe now since nothing runs and there are no consumers. Use `constants.py` everywhere.
11. Replace `print` debugging with `logging` levels.
12. Consolidate the three report files into one short `docs/` note (or delete; git history preserves them). Keep this report current or drop it.
13. Rewrite README: "Current status" section (what works/what doesn't), accurate Usage, real architecture diagram, Python/TensorFlow compatibility matrix, license.
14. Add `LICENSE` (MIT/Apache-2.0 if you're happy with permissive).

### P3 - project direction
15. Implement real Tarneeb rules: team scoring, auction with trump chosen by highest bidder, correct trump-selection or drop that rule from README.
16. Decide the ML approach: given "reinforcement learning" as the objective, use self-play with a policy/value net (with legal-move masking) instead of imitation of random play; use the existing `Turn.loss` idea as a reward shaping only if justified.
17. Seed RNGs, save/load models, log training metrics (TensorBoard).

---

## 7. Suggested Quick Wins Checklist

- [X] `git rm --cached Tarneeb/__pycache__/*.pyc`
- [X] Delete `neural-network-tricks-tarneeb/` nested folder
- [ ] Restore `TarneebPlayer`/`Turn` method bodies
- [ ] `if __name__ == "__main__"` in `GTarneeb.py`, remove `exit()`
- [ ] Add `LICENSE`, `tests/`, CI workflow
- [ ] Update README status and usage instructions
- [ ] Merge/delete the three evaluation documents

---

*Caveat: findings in §2 were verified by executing the code; findings in §3-§5 come from reading the source. No files other than this report were modified.*
