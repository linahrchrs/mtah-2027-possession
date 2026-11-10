"""
tune_external.py - tune possession.py's PARAMS on EXTERNAL data only.

Data: Metrica Sports public sample data, games 1 and 2
      https://github.com/metrica-sports/sample-data  (credit: Metrica Sports)
Metric: the official evaluation/possession_metrics.py (spellF1 ranks, macroF1 tie-breaks)

Nothing from the competition match (BAR-ATH) is read here: no ground truth, no
tracking, no video. That is what the rules require for tuning.

Usage (from the hackathon repo root):
    python tune_external.py --sample-dir C:/path/to/sample-data/data
It writes tuned_params.json next to possession.py; possession.py picks it up
automatically on its next run.

Method
  1. Reference labels from the sample EVENT data (human-coded, external):
     - dead ball: from a BALL OUT / goal / out-of-play shot / foul / offside
       until the next SET PIECE
     - possession: the team of the last on-ball event (pass, set piece, shot,
       recovery, ball lost); challenges do not change possession
  2. Degrade the sample tracking to resemble the competition's automatic
     tracking (numbers from DATA.md, not from any ground truth): ball removed
     in bursts during live play, position noise, player tracks dropped
     (~20.5 of 22 visible), Smart-Tagging-like restarts (throw-ins have no team).
  3. Coordinate descent over PARAMS, tuned on game 1, checked on game 2.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "evaluation"))

import possession as pz                       # noqa: E402
import possession_metrics as pm               # noqa: E402

FPS = pz.FPS
L, W = 105.0, 68.0                            # sample-data pitch (README)

GRID = dict(
    control_radius=[0.75, 1.0, 1.3, 1.6, 2.0, 2.5],
    rel_speed_max=[1.5, 3.0, 5.0, 8.0, 12.0, 99.0],
    control_frames=[2, 3, 4, 6, 10, 15],
    max_backdate=[25, 50, 75, 100],
    still_rel=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    still_frames=[25, 50, 100],
    walkback_rel=[1.0, 1.25, 1.5, 2.0],
    restart_lookback=[250, 500, 750],
    min_spell_frames=[12, 25, 50, 75],
    min_dead_frames=[12, 25, 50],
)


# --------------------------------------------------------------------------- load sample data
def load_tracking(sample_dir, g):
    frames = []
    for side in ("Home", "Away"):
        f = os.path.join(sample_dir, f"Sample_Game_{g}", f"Sample_Game_{g}_RawTrackingData_{side}_Team.csv")
        frames.append(pd.read_csv(f, skiprows=2))
    home, away = frames
    period = home["Period"].to_numpy(int)
    ball = home.iloc[:, -2:].to_numpy(float) * [L, W]

    def players_of(df):
        cols = list(df.columns)
        xs = [i for i, c in enumerate(cols) if c.startswith("Player")]
        return np.stack([df.iloc[:, [i, i + 1]].to_numpy(float) for i in xs], 1) * [L, W]

    ph, pa = players_of(home), players_of(away)
    players = np.concatenate([ph, pa], 1)
    team = np.r_[np.full(ph.shape[1], pz.HOME), np.full(pa.shape[1], pz.AWAY)]
    return period, ball, players, team


def reference_labels(sample_dir, g, n, period):
    """H/A/D/X codes per frame, derived from the human-coded sample events."""
    ev = pd.read_csv(os.path.join(sample_dir, f"Sample_Game_{g}", f"Sample_Game_{g}_RawEventsData.csv"))
    ev = ev.sort_values("Start Frame", kind="stable").reset_index(drop=True)
    typ = ev["Type"].str.upper()
    sub = ev["Subtype"].fillna("").str.upper()
    start = ev["Start Frame"].to_numpy(int) - 1
    end = np.where(ev["End Frame"] > 0, ev["End Frame"], ev["Start Frame"]).astype(int) - 1

    lab = np.full(n, pm.X)
    team_code = {"HOME": pm.H, "AWAY": pm.A}

    # possession: team of the last on-ball event
    on_ball = typ.isin(["PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT"])
    owner = np.full(n, -1)
    for i in np.flatnonzero(on_ball.to_numpy()):
        owner[start[i]] = team_code[ev.at[i, "Team"].upper()]
    owner = pd.Series(np.where(owner < 0, np.nan, owner)).ffill().to_numpy()

    # dead ball: stoppage event -> next restart. A restart is the next on-ball event,
    # not only a SET PIECE: in this event data goal kicks are coded as PASS/GOAL KICK.
    stop = (typ == "BALL OUT") | ((typ == "SHOT") & sub.str.contains("OUT|GOAL")) \
        | (sub.str.contains("FAULT") & ~sub.str.contains("ADVANTAGE")) \
        | (typ == "FAULT RECEIVED") | sub.str.contains("OFFSIDE")
    restarts = start[typ.isin(["PASS", "SET PIECE", "SHOT"]).to_numpy()]
    dead = np.zeros(n, bool)
    for i in np.flatnonzero(stop.to_numpy()):
        s = end[i] if typ[i] in ("BALL OUT", "SHOT") else start[i]
        nxt = restarts[restarts > s + 1]
        if len(nxt) and period[min(nxt[0], n - 1)] == period[min(s, n - 1)]:
            dead[s:nxt[0]] = True

    # graded span per period: first to last event
    for p in np.unique(period):
        ev_p = ev[ev["Period"] == p]
        a = int(ev_p["Start Frame"].min()) - 1
        b = int(max(ev_p["Start Frame"].max(), ev_p["End Frame"].max())) - 1
        seg = slice(a, b + 1)
        o = owner[seg]
        lab[seg] = np.where(np.isnan(o), pm.X, o).astype(int)
        lab[seg][dead[seg]] = pm.D
    return lab, ev


def sample_restarts(ev, rng):
    """Smart-Tagging-like restarts: set pieces, throw-ins without a team, 10% missed."""
    out = []
    for _, r in ev[ev["Type"].str.upper() == "SET PIECE"].iterrows():
        if rng.random() < 0.10:
            continue
        sub = str(r["Subtype"]).upper()
        t = pz.NONE if "THROW" in sub else (pz.HOME if r["Team"].upper() == "HOME" else pz.AWAY)
        jitter = int(rng.integers(-12, 13))                   # inferred tags are not frame-exact
        out.append((int(r["Start Frame"]) - 1 + jitter, t))
    return sorted(out)


# --------------------------------------------------------------------------- degrade
def bursts(n, frac, mean_len, rng, allowed=None):
    """Boolean mask covering ~frac of the allowed frames with bursts of geometric length."""
    mask = np.zeros(n, bool)
    allowed = np.ones(n, bool) if allowed is None else allowed
    target = frac * allowed.sum()
    while (mask & allowed).sum() < target:
        s = int(rng.integers(0, n))
        mask[s:s + int(rng.geometric(1 / mean_len))] = True
    return mask & allowed


def smooth_noise(shape, std_m, rng, span=25):
    """Position error that drifts over ~1 s, like a real tracker's, rather than
    jumping every frame (white noise would invent speeds of several m/s)."""
    w = rng.normal(0, 1, shape)
    k = np.ones(span) / span
    sm = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, w)
    return sm / sm.std() * std_m


def degrade(period, ball, players, team, labels, ev, alive_ball_miss, rng):
    n = len(ball)
    ball = ball + smooth_noise(ball.shape, 0.3, rng)
    alive = np.isin(labels, [pm.H, pm.A])
    ball[bursts(n, alive_ball_miss, 20, rng, alive)] = np.nan

    players = players + smooth_noise(players.shape, 0.2, rng)
    for k in range(players.shape[1]):                     # ~7% dropped -> ~20.5 of 22
        players[bursts(n, 0.07, 75, rng), k] = np.nan
    pspeed = np.linalg.norm(np.gradient(players, axis=0), axis=2) * FPS
    pspeed = pd.DataFrame(pspeed).rolling(5, center=True, min_periods=1).mean().to_numpy()

    halves = []
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        halves.append((int(idx[0]), int(idx[-1])))
    return dict(ball=ball, players=players, pspeed=pspeed, team=team,
                restarts=sample_restarts(ev, rng), halves=halves)


# --------------------------------------------------------------------------- score
CODE = {"H": pm.H, "A": pm.A, "D": pm.D, "X": pm.X}


def score(inp, gt, P):
    labels = pz.estimate(inp, P)
    pred = np.array([CODE[c] for c in labels])
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    spell = pm.spell_scores(gt, filled, runs)["mean_f1"]
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred)))
    return spell, macro


def mean_score(cases, P):
    s = np.array([score(inp, gt, P) for inp, gt in cases])
    return float(s[:, 0].mean()), float(s[:, 1].mean())


def coordinate_descent(cases, P0, rounds=2):
    P = dict(P0)
    best = mean_score(cases, P)
    print(f"  start                     spellF1 {best[0]:.4f}  macroF1 {best[1]:.4f}")
    for r in range(rounds):
        improved = False
        for name, values in GRID.items():
            for v in values:
                if v == P[name]:
                    continue
                trial = dict(P, **{name: v})
                s = mean_score(cases, trial)
                if s > best:                               # spellF1 first, macroF1 tie-break
                    best, P, improved = s, trial, True
                    print(f"  round {r + 1}: {name:<17}= {v:<5} spellF1 {s[0]:.4f}  macroF1 {s[1]:.4f}")
        if not improved:
            break
    return P, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True, help="the 'data' folder of metrica-sports/sample-data")
    ap.add_argument("--seeds", type=int, default=2, help="degradation draws per game")
    ap.add_argument("--out", default=os.path.join(HERE, "tuned_params.json"))
    args = ap.parse_args()

    games = {}
    for g in (1, 2):
        period, ball, players, team = load_tracking(args.sample_dir, g)
        gt, ev = reference_labels(args.sample_dir, g, len(ball), period)
        cases = []
        for seed in range(args.seeds):
            for miss in (0.20, 0.35):                      # live-play ball loss, two plausible levels
                rng = np.random.default_rng(1000 * g + 10 * seed + int(miss * 100))
                cases.append((degrade(period, ball, players, team, gt, ev, miss, rng), gt))
        n_poss = len(pm.possession_spells(gt, pm.valid_runs(gt)))
        print(f"game {g}: {len(ball)} frames, reference has {n_poss} possessions, {len(cases)} degraded copies")
        games[g] = cases

    P0 = dict(pz.PRIORS)          # always start from the hand-set priors, not a previous run
    print("\ntuning on game 1")
    P, _ = coordinate_descent(games[1], P0)

    print("\nhold-out check on game 2 (never used for tuning)")
    d0, d1 = mean_score(games[2], P0), mean_score(games[2], P)
    print(f"  defaults  spellF1 {d0[0]:.4f}  macroF1 {d0[1]:.4f}")
    print(f"  tuned     spellF1 {d1[0]:.4f}  macroF1 {d1[1]:.4f}")

    keep = d1[0] >= d0[0]
    final = P if keep else P0
    changed = {k: (P0[k], final[k]) for k in P0 if P0[k] != final[k]}
    print("\nkept tuned params" if keep else "\ntuned params did not generalise to game 2 -> keeping defaults")
    for k, (a, b) in changed.items():
        print(f"  {k:<17} {a} -> {b}")
    with open(args.out, "w") as f:
        json.dump({"params": final, "source": "Metrica Sports sample-data games 1-2 (external)",
                   "tuned_on": "game 1", "holdout": "game 2",
                   "holdout_spellF1": {"defaults": d0[0], "tuned": d1[0]}}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()