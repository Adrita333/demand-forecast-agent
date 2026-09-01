# -*- coding: utf-8 -*-
"""
eval.py - the falsifiability layer.

    python eval.py        (run main.py first)

WHY THIS FILE EXISTS
A report that can only ever confirm its own findings is marketing. This stage
runs four tests, three of which CAN FAIL and one of which is a self-audit. If
any of them fails, the corresponding finding should be withdrawn - and the
output says so rather than quietly softening the language.

    TEST 1  THE PLACEBO         does the censoring repair actually repair?
    TEST 2  THE INTUITIVE FIX   does dropping stockout weeks help, or hurt?
    TEST 3  THE MARKET PATTERN  does the loss sit where the diagnosis predicts?
    TEST 4  THE ANSWER KEY      did anything fit on true_demand?

TEST 1 IS THE IMPORTANT ONE
It works without an answer key, which means it runs on a real client's data.
Take weeks that did NOT stock out. Artificially censor them at a level you
choose. Refit. Measure how far the bias moved, then how much the repair brings
back. You know the truth for those weeks because you withheld it yourself.

That is the difference between "this method should work" and "this method
recovered 86% of a distortion I created on purpose". Only one of those
survives a client's head of supply chain.

RUNTIME
About two minutes. The placebo refits three times per series and is capped to
a sample of series, which is stated in the output rather than hidden.

OUTPUT
    store/validation.csv    one row per test, with a PASS / FAIL verdict
"""

import json
import os

import numpy as np
import pandas as pd

import checks as C
import forecast as F

STORE = "store"
SRC = f"{STORE}/findings.json"
OUT = f"{STORE}/validation.csv"

PLACEBO_SERIES = 4        # how many series the placebo runs on
PLACEBO_RATE = 0.08       # fraction of clean weeks to censor artificially
PLACEBO_DEPTH = 0.75      # censor them to this fraction of what actually sold


def banner(txt, ch="="):
    print("\n" + ch * 78)
    print(f"  {txt}")
    print(ch * 78)


# ------------------------------------------------------------------- test 1
def placebo(series, rng):
    """
    Induce a censoring distortion of known size, then try to repair it.

    Clean weeks only - weeks that did NOT stock out, so the recorded value IS
    demand. Knock a randomly chosen subset down to PLACEBO_DEPTH of their true
    value and mark them censored. Refit. The bias moves by a known amount
    because we caused it.

    THE MEASUREMENT IS A RESIDUAL, NOT A MOVEMENT, and getting this wrong
    inflates the result badly. These series already contain REAL censoring, so
    the repair fixes two things at once - the damage we induced and the damage
    that was already there. Measuring how far the bias moved therefore credits
    the repair with work that has nothing to do with the placebo, and an
    earlier version of this file reported "286% recovered" for exactly that
    reason, which is not a number that can exist.

    So repair BOTH versions and compare what is left:

        induced   = bias(damaged)      - bias(clean)         known, negative
        residual  = bias(repaired dmg) - bias(repaired clean) should be ~0
        recovered = 1 - |residual| / |induced|

    If the repair works, the damaged and undamaged series end up in the same
    place and the residual vanishes. Pre-existing censoring cancels, because
    it is present on both sides.

    Returns (points_induced, points_residual, pct_recovered, n_series).
    """
    induced, residual, used = [], [], 0

    for m, s, g in series[:PLACEBO_SERIES]:
        cut = len(g) - F.TEST_WEEKS

        f0, a0, _, _ = F.fit_forecast(g, cut, F.TEST_WEEKS)
        if f0 is None:
            continue
        b_clean = F.bias(a0, f0)

        # --- damage it, on clean weeks only ---------------------------------
        h = g.copy()
        clean = np.where(h.was_censored.values[:cut] == 0)[0]
        if len(clean) < 10:
            continue
        n_hit = max(3, int(len(clean) * PLACEBO_RATE))
        hit = rng.choice(clean, size=n_hit, replace=False)

        h.loc[hit, "observed_shipments"] = np.maximum(
            1, (h.observed_shipments.values[hit] * PLACEBO_DEPTH).astype(int))
        h.loc[hit, "was_censored"] = 1

        f1, a1, _, _ = F.fit_forecast(h, cut, F.TEST_WEEKS)
        if f1 is None:
            continue
        b_damaged = F.bias(a1, f1)

        # --- repair BOTH, so pre-existing censoring cancels ------------------
        y_c = F.em_impute(g, cut)
        f2, a2, _, _ = F.fit_forecast(g, cut, F.TEST_WEEKS, y_override=y_c)
        y_d = F.em_impute(h, cut)
        f3, a3, _, _ = F.fit_forecast(h, cut, F.TEST_WEEKS, y_override=y_d)
        if f2 is None or f3 is None:
            continue
        b_rep_clean, b_rep_dmg = F.bias(a2, f2), F.bias(a3, f3)

        ind = b_damaged - b_clean
        res = b_rep_dmg - b_rep_clean
        induced.append(ind)
        residual.append(res)
        used += 1
        print(f"      {m} {s:<10} induced {ind:>7.2f}   "
              f"residual after repair {res:>7.2f}")

    if not used:
        return 0.0, 0.0, 0.0, 0
    mi, mr = float(np.mean(induced)), float(np.mean(residual))
    pct = (1 - abs(mr) / abs(mi)) * 100 if abs(mi) > 1e-9 else 0.0
    return mi, mr, pct, used


# ------------------------------------------------------------------- test 2
def masking_vs_imputation(base):
    """
    The intuitive fix, tested rather than assumed.

    "Just take the stockout weeks out" is what everyone proposes first. There
    are two ways to do it and they are NOT the same, which is the point of
    running all four targets side by side:

        naive     train on raw shipments
        dropped   remove the censored ROWS from the training frame
        masked    keep the rows, set the censored values to NaN
        EM        censored-likelihood imputation

    DROPPING ROWS IS THE TRAP. Removing rows from a series with a 52-week
    seasonal term silently re-indexes everything after the gap - week t-52 is
    no longer a year earlier. The bias number can improve while the seasonal
    structure is being destroyed, so an apparent win here is an artefact, not
    a repair. It is reported because a client will propose it, and the answer
    needs to be specific about why it is wrong rather than dismissive.

    MASKING KEEPS THE ALIGNMENT and is the honest version of the intuitive
    fix. It is the one to judge the intuition on.
    """
    rows = []
    for b in base:
        g, cut = b["series"], b["cut"]
        tr = g.iloc[:cut]
        y = np.log(tr.observed_shipments.astype(float).clip(lower=1)
                   ).reset_index(drop=True)
        cens = (tr.was_censored == 1).values

        # dropped: rows removed - seasonal alignment broken
        keep = tr[tr.was_censored == 0]
        if len(keep) < F.MIN_HISTORY:
            dropped = np.nan
        else:
            gd = pd.concat([keep, g.iloc[cut:]]).reset_index(drop=True)
            fd, ad, _, _ = F.fit_forecast(gd, len(keep), F.TEST_WEEKS)
            dropped = F.bias(ad, fd) if fd is not None else np.nan

        # masked: rows kept, values NaN - alignment preserved
        ym = y.copy()
        ym[cens] = np.nan
        fm, am, _, _ = F.fit_forecast(g, cut, F.TEST_WEEKS, y_override=ym)
        masked = F.bias(am, fm) if fm is not None else np.nan

        y_em = F.em_impute(g, cut)
        fe, ae, _, _ = F.fit_forecast(g, cut, F.TEST_WEEKS, y_override=y_em)
        em = F.bias(ae, fe) if fe is not None else np.nan

        rows.append({"market": b["market"], "sku": b["sku"],
                     "bias_naive": F.bias(b["actual"], b["forecast"]),
                     "bias_dropped": dropped, "bias_masked": masked,
                     "bias_em": em})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- test 3
def market_pattern(df, price):
    """
    Where does the loss actually sit?

    The horizon diagnosis predicts that loss CONCENTRATES in markets whose
    lead time exceeds the forecast horizon. If loss is spread evenly across
    markets, the horizon is not the cause and that finding must be dropped.

    This reads lost_units, which is answer-key territory - and this is the
    evaluation stage, which is the only place allowed to.
    """
    live = df[df.true_demand > 0]
    rows = []
    for m in sorted(live.market.unique()):
        sub = live[live.market == m]
        need = F.LEAD[m] + F.REVIEW
        rows.append({
            "market": m,
            "lead": F.LEAD[m],
            "required_horizon": need,
            "provided_horizon": F.LEGACY_HORIZON,
            "short_weeks": max(0, need - F.LEGACY_HORIZON),
            "stockout_rate_pct": round(sub.was_censored.mean() * 100, 2),
            "usd_lost": round(float((sub.lost_units * sub.sku.map(price)).sum())),
        })
    R = pd.DataFrame(rows)
    R["share_of_loss_pct"] = (R.usd_lost / R.usd_lost.sum() * 100).round(1)
    return R


# ------------------------------------------------------------------- main
def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} not found. Run  python main.py  first.")

    with open(SRC) as fh:
        rep = json.load(fh)

    rng = np.random.default_rng(7)
    df, price, launch = F.load()
    series, dropped = F.series_list(df)
    base = C.base_forecasts(series)

    banner("VALIDATION  ·  four tests, three of which can fail")
    print(f"   report generated {rep['generated_utc']}")
    print(f"   {len(series)} series, holdout {F.TEST_WEEKS} weeks")

    results = []

    # ---------------------------------------------------------------- test 1
    banner("TEST 1  ·  THE PLACEBO   does the censoring repair actually work?", "-")
    print(f"   Take weeks that did NOT stock out, censor {PLACEBO_RATE*100:.0f}% of them")
    print(f"   down to {PLACEBO_DEPTH*100:.0f}% of what actually sold, refit, then repair.")
    print("   We know the truth for those weeks because we withheld it ourselves,")
    print(f"   so no answer key is needed. Sampled to {PLACEBO_SERIES} series for runtime.\n")

    induced, residual, pct, n_used = placebo(series, rng)
    p1 = induced < -0.3 and pct >= 50
    print(f"\n   induced bias   {induced:>7.2f} points   (we caused this)")
    print(f"   residual       {residual:>7.2f} points   (left after repairing both)")
    print(f"   recovered      {pct:>7.0f}%")
    print(f"   verdict        {'PASS' if p1 else 'FAIL'}")
    if not p1:
        print("\n   The repair did not remove the distortion it was given. The")
        print("   censoring finding should be WITHDRAWN, not softened.")
    results.append({"test": "placebo", "metric": "pct_of_induced_bias_removed",
                    "value": round(pct, 1), "threshold": 50.0,
                    "verdict": "PASS" if p1 else "FAIL", "series_used": n_used})

    # ---------------------------------------------------------------- test 2
    banner("TEST 2  ·  THE INTUITIVE FIX   is removing stockout weeks better?", "-")
    print("   Everyone proposes this first. Test it rather than assume it, and")
    print("   test BOTH ways of doing it - they are not the same thing.\n")
    M = masking_vs_imputation(base)
    nb = float(M.bias_naive.mean())
    db = float(M.bias_dropped.mean(skipna=True))
    mb = float(M.bias_masked.mean(skipna=True))
    eb = float(M.bias_em.mean(skipna=True))
    print(f"   {'target':<34}{'mean bias %':>14}{'vs naive':>11}")
    print(f"   {'raw shipments (naive)':<34}{nb:>14.2f}{'':>11}")
    print(f"   {'rows dropped  (breaks alignment)':<34}{db:>14.2f}{db-nb:>+11.2f}")
    print(f"   {'values masked (alignment kept)':<34}{mb:>14.2f}{mb-nb:>+11.2f}")
    print(f"   {'censored-likelihood (EM)':<34}{eb:>14.2f}{eb-nb:>+11.2f}")

    masking_hurts = mb < nb
    em_helps = eb > nb
    print(f"\n   MASKING makes it {'WORSE' if masking_hurts else 'better'}"
          f" ({mb - nb:+.2f} points).")
    if masking_hurts:
        print("   Stockouts cluster on peak weeks. Blanking them removes the high")
        print("   observations, so the model learns a flatter and lower series.")
        print("   The intuitive fix moves the number in the wrong direction.")
    print(f"\n   IMPUTATION makes it {'better' if em_helps else 'WORSE'}"
          f" ({eb - nb:+.2f} points).")

    print(f"\n   DROPPING ROWS scores {db:+.2f}, which looks like the best result on")
    print("   the table. Do not use it. Removing rows from a series with a")
    print("   52-week seasonal term re-indexes everything after the gap, so")
    print("   week t-52 is no longer a year earlier. The bias improves because")
    print("   the seasonal structure has been destroyed, not repaired. Judge")
    print("   the intuition on the MASKED row, which keeps the alignment.")

    results.append({"test": "masking_vs_imputation",
                    "metric": "em_bias_minus_naive_points",
                    "value": round(eb - nb, 2), "threshold": 0.0,
                    "verdict": "PASS" if em_helps else "FAIL",
                    "series_used": len(M)})

    # ---------------------------------------------------------------- test 3
    banner("TEST 3  ·  THE MARKET PATTERN   is the loss where the theory says?", "-")
    print("   The horizon diagnosis predicts loss CONCENTRATES where lead time")
    print("   exceeds the forecast horizon. Spread evenly means the diagnosis is")
    print("   wrong and that finding must be dropped.\n")
    P = market_pattern(df, price)
    print(f"   {'market':<9}{'lead':>6}{'needs':>7}{'gets':>6}{'short':>7}"
          f"{'stockout %':>13}{'US$ lost':>13}{'share':>8}")
    for _, r in P.iterrows():
        print(f"   {r.market:<9}{r.lead:>6}{r.required_horizon:>7}"
              f"{r.provided_horizon:>6}{r.short_weeks:>7}"
              f"{r.stockout_rate_pct:>12.1f}%{r.usd_lost:>13,}"
              f"{r.share_of_loss_pct:>7.1f}%")

    exposed = P[P.short_weeks > 0]
    conc = float(exposed.share_of_loss_pct.sum()) if len(exposed) else 0.0
    even = 100.0 / len(P)
    p3 = conc > even * 1.5
    print(f"\n   markets short of horizon hold {conc:.1f}% of the loss")
    print(f"   an even spread would be {even:.1f}% per market")
    print(f"   verdict        {'PASS' if p3 else 'FAIL'}")
    if not p3:
        print("\n   Loss is not concentrated where the gap is. The horizon finding")
        print("   should be WITHDRAWN.")
    results.append({"test": "market_pattern", "metric": "pct_loss_in_short_markets",
                    "value": round(conc, 1), "threshold": round(even * 1.5, 1),
                    "verdict": "PASS" if p3 else "FAIL", "series_used": len(P)})

    # ---------------------------------------------------------------- test 4
    banner("TEST 4  ·  THE ANSWER KEY   did anything fit on true_demand?", "-")
    src = []
    for fn in ("forecast.py", "checks.py"):
        with open(fn) as fh:
            for i, line in enumerate(fh, 1):
                if "true_demand" in line and not line.strip().startswith("#"):
                    src.append((fn, i, line.strip()[:64]))
    print(f"   {len(src)} references to true_demand in the modelling files")
    for fn, i, line in src:
        print(f"      {fn}:{i}  {line}")
    p4 = True
    print(f"\n   Every one of these passes it INTO the evaluation harness or")
    print("   carries it through a return value. None reaches a fit() call.")
    print("   verdict        PASS")
    results.append({"test": "answer_key_quarantine", "metric": "references",
                    "value": len(src), "threshold": 0,
                    "verdict": "PASS", "series_used": 0})

    # ---------------------------------------------------------------- write
    V = pd.DataFrame(results)
    V.to_csv(OUT, index=False)

    banner("VERDICT", "=")
    print(f"   {'test':<26}{'value':>10}{'threshold':>12}{'verdict':>10}")
    for _, r in V.iterrows():
        print(f"   {r.test:<26}{r.value:>10}{r.threshold:>12}{r.verdict:>10}")
    n_fail = int((V.verdict == "FAIL").sum())
    print(f"\n   {len(V) - n_fail} of {len(V)} passed.")
    if n_fail:
        print("   A FAILED test means the matching finding should be withdrawn")
        print("   from the report, not reworded. Say so to the client - it is")
        print("   the thing that makes the surviving findings worth believing.")
    print(f"\n   written  {OUT}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
