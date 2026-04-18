# Backlog

## 6. Tournament.run()
- Loop `tick()` until `active_teams` is empty (all teams have 3 wins or 3 losses).
- Return `self` for chaining / inspection.
- Guard against exceeding `round_strategies` length.

## 7. SwissMatchupStrategy
Single strategy that handles all swiss rounds internally. Receives all teams + history.

**Round 1:** Fixed pairings — 1v9, 2v10, ..., 8v16 (same as `PureSeedMatchup`).

**Rounds 2-3:** Group teams by W-L record. Within each pool, pair highest seed vs lowest seed, skipping rematches.

**Rounds 4-5:** Group teams by W-L record. Each pool should have exactly 6 teams. Use the hardcoded priority table (15 rows of 3 pairings). Pick the top-most row that produces no rematches within the stage.

Priority table (1-indexed by seed within the pool):
```
 1: 1v6 2v5 3v4
 2: 1v6 2v4 3v5
 3: 1v5 2v6 3v4
 4: 1v5 2v4 3v6
 5: 1v4 2v6 3v5
 6: 1v4 2v5 3v6
 7: 1v6 2v3 4v5
 8: 1v3 2v6 4v5
 9: 1v6 2v3 4v5  (verify exact table from rulebook)
10-15: remaining permutations
```
> TODO: transcribe the full 15-row table from the rulebook exactly.

**All rounds:** Set `best_of=3` for elimination matches (pool X-2 or 2-X) and advancement matches (pool 2-X or X-2). Set `best_of=1` otherwise.

**Rematch avoidance:** Check `history` for prior matchups between the same two teams.

## 8. Team.initial_seed
Already done in step 1 (added `initial_seed` field with auto-population from `seed`).

## 9. External orchestrator script
A simple script (e.g., `src/joust/__main__.py` or `scripts/simulate_major.py`) that:

1. Creates 16 teams with VRS-based seeds for Stage 1.
2. Runs Stage 1 `Tournament` with `SwissMatchupStrategy`.
3. Extracts `tournament.advanced` (3-win teams), assigns them seeds 9-16 based on final seed order.
4. Creates Stage 2 with 8 directly-invited teams (seeds 1-8) + 8 advancers (9-16).
5. Runs Stage 2.
6. Same for Stage 3.
7. Extracts top 8 from Stage 3, creates playoffs `Tournament` with `SingleEliminationStrategy`.
8. Runs playoffs.
9. Prints final rankings using the cascading tiebreaker (playoff position > stage 3 record > stage 3 difficulty > ... > initial seed).

## 10. SingleEliminationStrategy
For playoffs. All BO3.

**Round 1 bracket:**
- Bracket A: 1v8, 4v5
- Bracket B: 2v7, 3v6

**Semifinals:** Winners of 1v8 vs 4v5, winners of 2v7 vs 3v6.

**Grand final:** Semifinal winners.

Implementation: track bracket position via match history. Each call to `get_matchups` pairs winners from the previous round. The strategy needs to maintain bracket structure (A vs A, B vs B until the final).
