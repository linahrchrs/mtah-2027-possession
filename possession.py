"""
possession.py - rule-based ball-possession estimator for More than a Hack 2027 (Phase 1).

Run from the root of the cloned repo, with the data unpacked in ./data:
    python possession.py --out evaluation/results/rules_v1.csv
    python evaluation/validate_submission.py evaluation/results/rules_v1.csv

Nothing here is trained. Every threshold in PARAMS is a physical prior
(documented next to it). Per the rules, if you tune them, tune ONLY on
external data (e.g. Metrica's public sample games) - never on the public
5-minute ground truth or the provided video.

Pipeline
  1. load tracking (kloppy) -> ball (N,2) and players (N,P,2) in metres, team per slot
  2. clean the ball: interpolate short gaps only
  3. per-frame "control candidate": nearest player within a radius, ball speed
     close to the player's speed (control, not a rebound), and no opponent
     equally close (contested balls change nothing)
  4. hysteresis state machine: a switch is confirmed only after the other team
     keeps control for `control_frames` frames, then dated back to that team's
     first touch (mid-gap if the ball was invisible); loose ball and missing
     ball keep the previous team (matches the official definition)
  5. dead ball: ball out of bounds, or everyone standing still, or the
     stoppage before a Smart-Tagging set piece; the restarting team takes over
  6. clean-up for spellF1: absorb very short possession / dead-ball spells
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FPS = 25
N_FRAMES = 199_319
PITCH_L, PITCH_W = 105.3, 68.0
HALVES = [(11185, 90651), (110808, 190197)]  # graded video frames, from DATA.md

PARAMS = dict(
    ball_gap_interp=10,     # frames (0.4 s): only bridge short ball dropouts
    control_radius=1.6,     # m: typical reach of a player controlling the ball
    rel_speed_max=3.0,      # m/s: ball moving with the player = control, else rebound/pass
    control_frames=6,       # frames (0.24 s) of sustained control to change possession
    max_backdate=50,        # frames (2 s): a switch is dated back at most this far (a pass in flight)
    out_margin=0.75,        # m beyond the lines before we call the ball out (tracking noise)
    still_rel=0.6,          # play is stopped when mean player speed < this x the match's
                            # typical (median) in-play speed - scale-free, so it transfers
                            # between trackers that measure speed differently
    still_frames=50,        # frames (2 s) of stillness to call a stoppage
    walkback_rel=1.25,      # before a set piece, a lost ball + this x stillness speed = still stopped
    restart_lookback=750,   # frames (30 s): max dead-ball stretch before a set piece
    min_spell_frames=25,    # frames (1 s): shorter possession spells are absorbed
    min_dead_frames=25,     # frames (1 s): shorter dead-ball spells are absorbed
)

PRIORS = dict(PARAMS)  # hand-set physical priors, kept for tune_external.py

# Settings tuned on EXTERNAL data by tune_external.py override the priors above.
_TUNED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned_params.json")
if os.path.exists(_TUNED):
    import json
    with open(_TUNED) as _f:
        PARAMS.update(json.load(_f)["params"])

HOME, AWAY, NONE = 0, 1, -1


# --------------------------------------------------------------------------- loading
def load_inputs():
    """Return dict with ball (N,2), players (N,P,2), pspeed (N,P), team (P,), restarts list."""
    from kloppy import metrica
    from games import asset_paths

    paths = asset_paths()
    ds = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
    df = ds.to_df()
    df["video_frame"] = df["frame_id"].astype(int) - 1          # DATA.md: frame_id is 1-based
    df = df.set_index("video_frame").reindex(range(N_FRAMES))

    slots, team = [], []
    for t in ds.metadata.teams:
        is_home = "home" in str(t.ground).lower()
        for pl in t.players:
            if f"{pl.player_id}_x" in df.columns:
                slots.append(pl.player_id)
                team.append(HOME if is_home else AWAY)
    if not slots:
        raise SystemExit("No player columns matched - print(df.columns[:20]) and adapt load_inputs().")
    print(f"{len(slots)} track slots: {team.count(HOME)} home, {team.count(AWAY)} away")

    ball = np.c_[df["ball_x"].to_numpy(float) * PITCH_L, df["ball_y"].to_numpy(float) * PITCH_W]
    px = np.stack([df[f"{s}_x"].to_numpy(float) for s in slots], 1) * PITCH_L
    py = np.stack([df[f"{s}_y"].to_numpy(float) for s in slots], 1) * PITCH_W
    players = np.stack([px, py], 2)
    sp_cols = [f"{s}_s" for s in slots]
    if all(c in df.columns for c in sp_cols):
        pspeed = np.stack([df[c].to_numpy(float) for c in sp_cols], 1)
    else:  # derive speed from positions
        pspeed = np.linalg.norm(np.gradient(players, axis=0), axis=2) * FPS

    return dict(ball=ball, players=players, pspeed=pspeed,
                team=np.array(team), restarts=load_restarts())


def load_restarts():
    """Set pieces from Smart Tagging -> list of (video_frame, team or NONE)."""
    try:
        from kloppy import sportscode
        from games import asset_paths
        ev = sportscode.load(asset_paths()["smart_tagging"]).to_df()
    except Exception as e:  # tagging is optional evidence
        print(f"Smart Tagging not used ({e})")
        return []
    ev = ev[ev["code"].str.upper() == "SET PIECES"]
    out = []
    for _, r in ev.iterrows():
        fr = int(round(r["timestamp"].total_seconds() * FPS))
        tag = str(r.get("Team", "")).lower()
        t = HOME if "barcelona" in tag else AWAY if "atletic" in tag or "athletic" in tag else NONE
        out.append((fr, t))
    print(f"{len(out)} set pieces from Smart Tagging")
    return sorted(out)


# --------------------------------------------------------------------------- helpers
def interp_short_gaps(x, max_gap):
    """Linear interpolation of NaN runs no longer than max_gap (per column)."""
    s = pd.DataFrame(x)
    filled = s.interpolate(limit=max_gap, limit_area="inside")
    # pandas `limit` fills the first max_gap of a long gap too: undo that
    isna = s.isna().all(axis=1).to_numpy()
    run_id = np.cumsum(~isna)
    run_len = pd.Series(isna).groupby(run_id).transform("sum").to_numpy()
    out = filled.to_numpy(copy=True)
    out[isna & (run_len > max_gap)] = np.nan
    return out


def runs(labels):
    """Yield (start, end_exclusive, value) for runs of equal values."""
    n = len(labels)
    if n == 0:
        return
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]
    for s, e in zip(starts, ends):
        yield s, e, labels[s]


def rolling_mean(x, w):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


# --------------------------------------------------------------------------- core
def control_candidates(ball, players, pspeed, team, P):
    """Per frame: team clearly controlling the ball, or NONE."""
    present = ~np.isnan(ball).any(1)
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)

    j = np.argmin(d, axis=1)
    dmin = d[np.arange(len(d)), j]
    d_home = np.where(team == HOME, d, np.inf).min(1)
    d_away = np.where(team == AWAY, d, np.inf).min(1)
    contested = (d_home < P["control_radius"]) & (d_away < P["control_radius"])

    bspeed = np.linalg.norm(np.gradient(ball, axis=0), axis=1) * FPS
    bspeed = rolling_mean(bspeed, 5)
    ps = pspeed[np.arange(len(d)), j]
    moving_with = np.abs(bspeed - np.nan_to_num(ps, nan=99)) < P["rel_speed_max"]

    ok = present & (dmin < P["control_radius"]) & moving_with & ~contested
    cand = np.where(ok, team[j], NONE)
    # first-touch evidence: ball within reach of one team only, speed not required
    # (a reception decelerates the ball, so the speed test fails on the first frames)
    near_ok = present & (dmin < P["control_radius"]) & ~contested
    near = np.where(near_ok, team[j], NONE)
    return cand, near


def find_switch_frame(new, t, last_own, near, cand, missing, P):
    """Frame where the new team took the ball, for a switch confirmed at frame t.

    Official rule: loose ball stays with the old team, possession changes when the
    new team gets control. So the change is the new team's first touch after the
    old team's last control - not the moment our evidence became sufficient.
    """
    lo = max(last_own + 1, t - P["max_backdate"], 0)
    window = np.flatnonzero((near[lo:t + 1] == new) | (cand[lo:t + 1] == new))
    s = lo + window[0] if len(window) else t
    # ball invisible right before that first touch: the reception happened somewhere
    # inside the gap; its midpoint minimises the expected timing error
    g = s
    while g > lo and missing[g - 1]:
        g -= 1
    if g < s:
        s = (g + s) // 2
    return s


def possession_track(cand, near, missing, P, start_team=NONE):
    """Hysteresis: confirm a switch after `control_frames` frames of the other team's
    control, then date it back to that team's first touch."""
    n = len(cand)
    out = np.full(n, NONE)
    cur, streak_team, streak, last_own = start_team, NONE, 0, -1
    for t in range(n):
        c = cand[t]
        if c != NONE and c != cur:
            streak = streak + 1 if c == streak_team else 1
            streak_team = c
            if cur == NONE:
                cur, streak, last_own = c, 0, t
            elif streak >= P["control_frames"]:
                s = find_switch_frame(c, t, last_own, near, cand, missing, P)
                cur, streak, last_own = c, 0, t
                out[s:t + 1] = cur
        elif c == cur:
            streak, last_own = 0, t
        out[t] = cur
    return out


def dead_ball(ball, players, pspeed, team, restarts, P):
    """Boolean dead-ball mask + forced restart owners."""
    n = len(ball)
    m = P["out_margin"]
    x, y = ball[:, 0], ball[:, 1]
    out = (x < -m) | (x > PITCH_L + m) | (y < -m) | (y > PITCH_W + m)  # NaN -> False

    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # frames with no player tracked
            mean_speed = rolling_mean(np.nanmean(pspeed, axis=1), 25)
    mean_speed = np.nan_to_num(mean_speed, nan=0.0)
    # typical speed of THIS match, over frames where most players are tracked
    # (excludes pre-match / halftime, when the pitch is empty). No labels involved.
    tracked = (~np.isnan(pspeed)).sum(1) >= 15
    typical = np.median(mean_speed[tracked]) if tracked.any() else np.median(mean_speed)
    still_thr = P["still_rel"] * typical
    still = mean_speed < still_thr
    dead = np.zeros(n, bool)
    for s, e, v in runs(still):
        if v and e - s >= P["still_frames"]:
            dead[s:e] = True
    dead |= out

    # stoppage before each set piece: walk back while players stand still, the ball is
    # out, or the ball is lost and play is slow (walkback_rel x the stillness threshold)
    forced = {}
    ball_lost = np.isnan(ball).any(1)
    for fr, t in restarts:
        if not 0 < fr < n:
            continue
        s = fr
        lo = max(0, fr - P["restart_lookback"])
        while s > lo and (still[s - 1] or out[s - 1]
                          or (ball_lost[s - 1] and mean_speed[s - 1] < P["walkback_rel"] * still_thr)):
            s -= 1
        if fr - s >= P["min_dead_frames"]:
            dead[s:fr] = True
        dead[fr] = False  # restart frame itself is alive
        if t == NONE:  # throw-ins are N/A: the taker is the nearest player to the ball
            win = slice(fr, min(n, fr + FPS))
            d = np.linalg.norm(players[win] - ball[win][:, None, :], axis=2)
            if np.isfinite(d).any():
                t = team[np.unravel_index(np.nanargmin(d), d.shape)[1]]
        forced[fr] = t
    return dead, forced


def absorb_short(labels, value_set, min_len):
    """Replace short runs whose value is in value_set by their longer neighbour."""
    lab = labels.copy()
    for _ in range(5):
        changed = False
        rs = list(runs(lab))
        for i, (s, e, v) in enumerate(rs):
            if v in value_set and e - s < min_len:
                prev_ = rs[i - 1] if i > 0 else None
                next_ = rs[i + 1] if i + 1 < len(rs) else None
                if prev_ and next_ and prev_[2] == next_[2]:
                    lab[s:e] = prev_[2]
                elif prev_ or next_:
                    nb = max([r for r in (prev_, next_) if r], key=lambda r: r[1] - r[0])
                    lab[s:e] = nb[2]
                changed = True
        if not changed:
            break
    return lab


def estimate(inp, P=PARAMS):
    ball = interp_short_gaps(inp["ball"], P["ball_gap_interp"])
    players, pspeed, team = inp["players"], inp["pspeed"], inp["team"]

    cand, near = control_candidates(ball, players, pspeed, team, P)
    missing = np.isnan(ball).any(1)
    dead, forced = dead_ball(ball, players, pspeed, team, inp.get("restarts", []), P)

    # a restart forces the owner for a short window so the state machine starts right
    for fr, t in forced.items():
        if t != NONE:
            cand[fr:fr + P["control_frames"]] = t

    labels = np.full(len(ball), "X", dtype="<U1")
    halves = inp.get("halves") or (HALVES if len(ball) == N_FRAMES else [(0, len(ball) - 1)])
    for a, b in halves:
        seg = slice(a, b + 1)
        c, nr = cand[seg].copy(), near[seg].copy()
        c[dead[seg]] = NONE  # no possession evidence while dead
        nr[dead[seg]] = NONE
        poss = possession_track(c, nr, missing[seg], P)
        # before the first controlled touch: take the first known owner
        first = np.flatnonzero(poss != NONE)
        if len(first):
            poss[:first[0]] = poss[first[0]]
        lab = np.where(poss == HOME, "H", np.where(poss == AWAY, "A", "X"))
        lab[dead[seg]] = "D"
        lab = absorb_short(lab, {"D"}, P["min_dead_frames"])
        lab = absorb_short(lab, {"H", "A"}, P["min_spell_frames"])
        labels[seg] = lab
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/results/rules_v1.csv")
    args = ap.parse_args()

    if os.path.exists(_TUNED):
        print(f"using tuned params from {_TUNED}")
    labels = estimate(load_inputs())
    pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels}).to_csv(args.out, index=False)

    lab = pd.Series(labels)
    spells = sum(1 for _, _, v in runs(labels) if v in ("H", "A"))
    print(f"wrote {args.out}: {lab.value_counts().to_dict()}  possessions={spells}")


if __name__ == "__main__":
    main()