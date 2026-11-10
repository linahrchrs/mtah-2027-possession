# Ball possession from automatic tracking — technical document

**More than a Hack 2027, Phase 1** · Team: `lamasIA`
Repository: https://github.com/linahrchrs/mtah-2027-possession

## 1. Approach

The submission comes from a **rule-based, explainable possession tracker** over the automatic tracking data (ATD), with its thresholds **tuned only on external data**: Metrica Sports' public sample games. No model is trained, and nothing is fitted on the competition match, its ground truth or its video.

We chose rules over a learned model for three reasons:

- **Little usable training data.** The only labelled data we were allowed to fit on is external, and it comes from a different tracker. A model with many parameters would learn that tracker's quirks. A handful of physically meaningful thresholds transfers better.
- **The metric rewards structure.** spellF1 counts whole possessions, so the main risk is fragmentation. An explicit state machine with hysteresis controls fragmentation directly.
- **Explainability.** Every label can be traced back to a rule: who was nearest the ball, whether play had stopped, which restart followed.

### Pipeline (`possession.py`)

1. **Load** the tactical ATD with kloppy. Positions are converted to metres (105.3 × 68.0 m). Each of the 46 track slots is mapped to home or away using the metadata. Video frame = `frame_id − 1`.
2. **Clean the ball.** Only gaps of up to 10 frames (0.4 s) are interpolated. Longer gaps stay missing rather than being invented.
3. **Control evidence per frame.**
   - The nearest player must be within `control_radius` of the ball.
   - No opponent may be within the same radius; a contested ball gives no evidence.
   - A second, looser signal, *first-touch* evidence, records which team was within reach of the ball without any speed condition.
4. **Possession state machine.**
   - A change is *confirmed* only after `control_frames` frames of control by the other team.
   - It is then *dated back* to that team's first touch after the previous team's last control, at most `max_backdate` frames earlier.
   - If the ball was invisible just before that first touch, the change is placed at the middle of the gap, which minimises the expected timing error.
   - A loose or missing ball keeps the previous team.
5. **Dead-ball detection.**
   - The ball is outside the pitch by more than 0.75 m.
   - Or play is stopped: the mean player speed stays below `still_rel` × *this match's own median speed* for at least `still_frames` frames. The median is computed from the tracking itself (frames with at least 15 tracked players) and uses no labels.
   - Or there is a stoppage before a **Smart Tagging set piece**: walking back from the restart frame while players are still, the ball is out, or the ball is lost and play is slow.
6. **Restart owner.** This is the team tag of the Smart Tagging set piece. When the tag is `N/A` (every throw-in), it is the team of the player nearest the ball in the first second after the restart.
7. **Clean-up for spellF1.** Dead-ball runs shorter than `min_dead_frames` and possession runs shorter than `min_spell_frames` are absorbed by their neighbours.
8. **Output.** One label per video frame. `X` appears only on ungraded frames (pre-match, halftime, post-match), so every graded frame gets `H`, `A` or `D`.

## 2. Assets

| Asset | Used? | How |
|---|---|---|
| Tactical ATD (players + ball) | **Yes, main input** | positions, speeds (`_s`), team membership |
| Smart Tagging | **Yes** | `SET PIECES` only: restart frames, and restart team when tagged |
| Tactical homography | No | not needed: the ATD is already in pitch coordinates |
| Broadcast video / ATD / homography | No | not frame-aligned, and 34% of frames have no usable homography |
| Tactical video | No | not used as input |
| Public 5-minute ground truth | Correctness checks only | see §6 |
| **Metrica Sports sample-data, games 1–2** (external, public) | **Tuning** | see §5 |

Tried and dropped:

- **Absolute speed thresholds for "play stopped"** (v3). Tuned on the external data, a fixed 1.8 m/s did not transfer, because the competition ATD measures speed on a different scale. v3 labelled 58% of the graded frames dead, which is implausible. The match-relative threshold in v4 fixes this; it now labels 42%.
- **A ball-speed test for control** (ball moving at the player's speed). External tuning switched it off (`rel_speed_max` = 99): in this noisy tracking it rejected more real receptions than rebounds.

## 3. Possession rules

These follow CHALLENGE.md's operational definition.

- **Loose balls:** the team that last had control keeps possession, including while the ball is invisible.
- **Changes:** possession changes only when the other team *controls* the ball, meaning sustained proximity for `control_frames` frames with no opponent equally close. A single touch, a deflection, or a won aerial duel without follow-up control does not change possession, because it never reaches the confirmation threshold.
- **Timing:** once a change is confirmed, it is placed at the new team's first touch, not at the moment of confirmation.
- **Contested balls:** when players from both teams are within reach, the frame carries no evidence, and the previous owner is kept.
- **Restarts:** every restart has an owner, taken from the Smart Tagging team, or from the nearest player for throw-ins.
- **Dead ball:** labelled `D`, never owned (as the challenge requires).

## 4. Parameters

| Parameter | Prior (hand-set) | Tuned (external) | Meaning |
|---|---|---|---|
| `control_radius` | 1.6 m | **0.75 m** | reach for control |
| `rel_speed_max` | 3.0 m/s | **off (99)** | ball/player speed agreement |
| `control_frames` | 6 | **3** | frames of control to confirm a change |
| `max_backdate` | 50 | **100** | max frames a change is dated back |
| `still_rel` | 0.6 | **0.9** | "stopped" = below this × the match's median speed |
| `still_frames` | 50 | **100** | minimum stoppage length (frames) |
| `walkback_rel` | 1.25 | **2.0** | strictness of the walk-back before a set piece |
| `min_dead_frames` | 25 | **50** | shorter dead runs are absorbed |
| `min_spell_frames` | 25 | 25 | shorter possession runs are absorbed |
| `ball_gap_interp`, `out_margin`, `restart_lookback` | 10, 0.75 m, 750 | unchanged | |

## 5. External tuning (`tune_external.py`)

The rules require tuning on external data only. We used **Metrica Sports' public sample data, games 1 and 2** (github.com/metrica-sports/sample-data; credit: Metrica Sports).

1. **Reference labels from the human-coded sample events.**
   - Possession is the team of the last on-ball event (pass, set piece, shot, recovery, ball lost); challenges do not change it.
   - Dead ball runs from a stoppage (ball out, an out-of-play shot or goal, a foul, an offside) to the next on-ball restart. In this event data, goal kicks are coded as passes, not set pieces.
   - The reference has 37–45% dead ball and about 350 possessions per game.
2. **Degrading the sample tracking to resemble the competition ATD**, using figures from DATA.md, never ground truth:
   - the ball is removed in bursts during live play (20% and 35% levels)
   - position error drifts smoothly (0.3 m for the ball, 0.2 m for players)
   - about 7% of player tracks are dropped, leaving about 20.5 of 22 visible
   - restarts imitate Smart Tagging: throw-ins have no team, 10% are missed, and timing is jittered by ±0.5 s
3. **Coordinate descent** over the grid in `tune_external.py`, scored with the **official** `possession_metrics.py` (spellF1 first, macroF1 as tie-break). It tunes on game 1 and always starts from the priors.
4. **Hold-out check on game 2,** which is never used for tuning:

| Game 2 (held out) | spellF1 | macroF1 |
|---|---|---|
| priors | 0.316 | 0.448 |
| **tuned** | **0.501** | **0.764** |

The tuned parameters are kept only if they improve the held-out game. The result is written to `tuned_params.json`, which `possession.py` loads automatically.

## 6. Use of the public ground truth

The public 5 minutes were used only to check that the pipeline is correct: frame alignment, file format, and plausible counts. **No parameter was set by scoring against it.**

One design change was prompted by its report, and we disclose it:

- The v1 report showed possession changes about 16 frames late.
- The fix (dating changes back to the first touch) follows from the official definition of possession.
- It was validated on simulated matches, not by re-scoring the public window.

Final scores on that window, with 1,000-draw bootstrap intervals:

| Version | spellF1 | 90% interval | macroF1 |
|---|---|---|---|
| baseline_smooth50 (floor) | 0.326 | 0.21 – 0.45 | 0.621 |
| v2 (hand-set priors) | 0.423 | 0.26 – 0.57 | 0.819 |
| **v4 (submitted)** | **0.378** | 0.23 – 0.52 | 0.796 |

The scorer reports that gaps below 0.33 are noise on this window. We chose v4 over v2 because its parameters come from the external, held-out procedure, not because of this table.

On the full match, as a check that needs no labels, v4 labels 42% of graded frames dead and finds 344 possessions averaging 10.7 s. Both are realistic for a professional match.

## 7. Limitations

- **The simulation gap.** Tuning relies on degraded sample tracking. In the sample games the ball is missing during most stoppages, and v4 partly relies on that "ball lost before a set piece" signal. If the competition ATD keeps tracking the ball out of play, that part helps less.
- **Wrong team after receptions.** The tight 0.75 m radius increases home/away confusion on the public window (536 home frames labelled away). A radius scaled to local player density could reduce this.
- **No pass model.** A long pass to a teammate is never confused for a change, but an interception during a long ball-loss gap is placed by a rule (the middle of the gap), not estimated from the trajectory.
- **Identity switches and duplicates** are not repaired: team membership per slot is trusted.
- **With more time:**
  - model the ball trajectory through gaps (a Kalman filter or physics-based interpolation)
  - use the homography and video to recover the ball in the 41% of frames where the ATD loses it
  - tune on more external games, with more realistic tracker noise

## 8. Reproduction

```bash
git clone https://github.com/linahrchrs/mtah-2027-possession.git
cd mtah-2027-possession
pip install -r requirements-solution.txt
# unpack the official match data into ./data (not redistributable, not in the repo)
python possession.py --out lamasIA.csv
python evaluation/validate_submission.py lamasIA.csv
```

`possession.py` is deterministic, so this regenerates the submitted file exactly, using `tuned_params.json` from the repo.

To reproduce the tuning (optional, about 1 minute):

```bash
git clone https://github.com/metrica-sports/sample-data.git ../sample-data
python tune_external.py --sample-dir ../sample-data/data
```