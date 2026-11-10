# Ball possession from automatic tracking — More than a Hack 2027

Our solution to **Phase 1 of [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)** (Metrica Sports · FC Barcelona · Mobile World Capital Barcelona): for every frame of FC Barcelona vs Athletic Club, label who has the ball (`H` / `A`), dead ball (`D`), or no answer on ungraded frames (`X`).

It's a **rule-based possession tracker** over the automatic tracking data. Its few thresholds are **tuned only on external data**: Metrica's public sample games, checked on a held-out game. Nothing is trained or tuned on the competition match, its ground truth or its video.

Full write-up: **[TECHNICAL.md](TECHNICAL.md)**.

## How it works

1. Load the tactical tracking (players and ball, in metres) and the team of each track slot.
2. **Control:** the nearest player within reach of the ball, with no opponent equally close.
3. **State machine:** a change is confirmed after a few frames of control by the other team, then dated back to that team's first touch. A loose or invisible ball keeps the previous owner.
4. **Dead ball:** the ball is out of bounds, players are standing still relative to this match's typical speed, or there is a stoppage before a Smart Tagging set piece. The restarting team then takes the ball.
5. **Clean-up:** very short possessions and stoppages are absorbed, since the ranking metric (spellF1) penalises fragmented possessions.

## Run it

Requires Python 3.9+.

```bash
pip install -r requirements-solution.txt
```

Unpack the official match data into `./data`. It is licensed for this competition only and is **not** included here. The layout is described in [DATA.md](DATA.md).

```bash
python possession.py --out lamasIA.csv                # regenerates the submission (deterministic)
python evaluation/validate_submission.py lamasIA.csv   # official format check
```

`possession.py` loads `tuned_params.json` automatically. To regenerate that file from external data (about 1 minute):

```bash
git clone https://github.com/metrica-sports/sample-data.git ../sample-data
python tune_external.py --sample-dir ../sample-data/data
```

## Files

| File | Purpose |
|---|---|
| `possession.py` | the possession estimator; writes the submission CSV |
| `tune_external.py` | tunes `possession.py`'s thresholds on Metrica's public sample games |
| `tuned_params.json` | the tuned thresholds used for the submission |
| `TECHNICAL.md` | technical document: approach, assets, rules, limitations, reproduction |
| `evaluation/`, `starter/`, `games.py` | official scorer and starter code from the challenge repository (MIT) |

## Credits

- Challenge, scorer and starter code: [metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027), MIT licence.
- Tuning data: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data), credit Metrica Sports.
- Match data: provided to enrolled participants under the competition's terms. Not redistributed here.