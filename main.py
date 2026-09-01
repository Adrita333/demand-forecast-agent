# -*- coding: utf-8 -*-
"""
main.py - run the health check.

    python main.py

WHAT HAPPENS HERE
One pass over the client's demand history:

    load  ->  select fittable series  ->  fit once per series
          ->  run five gates  ->  rank  ->  write store/findings.json

The fitting happens ONCE, in base_forecasts(), and all the gates share it. That
is deliberate: if each gate refitted, two gates could disagree about the same
series and the comparison between them would stop meaning anything.

Expect roughly two to three minutes. Most of it is SARIMAX with a 52-week
seasonal term, refitted at four rolling origins per series for the horizon
gate, plus three EM rounds per series for the censoring gate.

WHAT IT DOES NOT DO
It does not recommend a model. Nothing in this report says "use a neural
network". Every finding is a fault in the process around the forecast -
the metric, the buffer, the training data, the horizon, the coverage rule -
because that is where the money nearly always is.

OUTPUT
    store/findings.json    the full findings, evidence included

Run store.py next to flatten that into the CSVs the app reads.
"""

import json
import os
import time

import numpy as np

import checks as C
import forecast as F

STORE = "store"
OUT = f"{STORE}/findings.json"


def _jsonable(o):
    """numpy types do not serialise; convert on the way out."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def banner(txt, ch="="):
    print("\n" + ch * 78)
    print(f"  {txt}")
    print(ch * 78)


def main():
    os.makedirs(STORE, exist_ok=True)
    t0 = time.time()

    df, price, launch = F.load()
    series, dropped = F.series_list(df)

    banner("DEMAND FORECAST HEALTH CHECK")
    print(f"   {df.week.nunique()} weeks  ·  {df.market.nunique()} markets  "
          f"·  {df.sku.nunique()} SKUs")
    print(f"   {len(series)} series fittable, {len(dropped)} excluded for "
          f"insufficient history")
    print(f"   model: SARIMAX{F.ORDER}{F.SEASONAL} on log(shipments), "
          f"exog = {', '.join(F.EXOG)}")
    print(f"   holdout: last {F.TEST_WEEKS} weeks, never seen during fitting")

    print("\n   fitting one model per series ...", end=" ", flush=True)
    base = C.base_forecasts(series)
    print(f"{len(base)} fitted in {time.time() - t0:.0f}s")

    findings = []
    for label, fn in [
        ("metric", lambda: C.check_metric(base, price)),
        ("safety stock", lambda: C.check_safety_stock(base, price)),
        ("censored history", lambda: C.check_censoring(series, price, base)),
        ("horizon", lambda: C.check_horizon(series, base, price)),
        ("coverage", lambda: C.check_coverage(series, dropped, df)),
    ]:
        t = time.time()
        print(f"   gate: {label:<20}", end="", flush=True)
        f = fn()
        findings.append(f)
        print(f"{f['severity']:<8}{time.time() - t:>6.0f}s")

    findings = C.rank(findings)

    # ------------------------------------------------------------- the report
    banner("FINDINGS, RANKED", "=")
    print("   Ranked by severity first, then by recoverable share. NOT by money")
    print("   alone - the cheapest fixes here are parameters, and a client who")
    print("   sees a parameter change at the top will do it this week.\n")
    print(f"   {'':<3}{'gate':<22}{'severity':<10}{'recovers':>10}   fix")
    for i, f in enumerate(findings, 1):
        print(f"   {i:<3}{f['id']:<22}{f['severity']:<10}"
              f"{f['recoverable_share']*100:>9.1f}%   {f['cost_to_fix'][:38]}")

    for i, f in enumerate(findings, 1):
        banner(f"{i}  {f['id']}  ·  {f['severity']}", "-")
        print(f"   {f['title']}\n")
        for line in _wrap(f["diagnosis"]):
            print(f"   {line}")
        print(f"\n   RECOVERS   {f['recoverable_share']*100:.1f}% of the "
              "stockout loss measured on the holdout")
        print(f"   COST       {f['cost_to_fix']}")
        print("\n   HOW TO PROVE THIS WRONG")
        for line in _wrap(f["validation"]):
            print(f"      {line}")
        print("\n   WHAT THIS DOES NOT ESTABLISH")
        for line in _wrap(f["caveat"]):
            print(f"      {line}")

    # ------------------------------------------------------ the honest total
    banner("WHY THESE DO NOT ADD UP", "-")
    print("   The shares OVERLAP and must not be summed.\n")
    print("   · censored data biases the forecast down, and the metric then")
    print("     rewards pushing it down further. Fixing either helps; fixing")
    print("     both is not the sum of the two.")
    print("   · a properly sized buffer absorbs most of the horizon shortfall,")
    print("     so gate 2 largely contains gate 4. See z_sensitivity in the")
    print("     horizon evidence, where the share falls to zero once z > 0.")
    print("\n   The defensible statement is the LARGEST single share, plus a")
    print("   qualitative note that the others compound it. Anyone adding these")
    print("   to reach a headline number is overselling, and the client's")
    print("   finance team will find it.")

    biggest = max(findings, key=lambda f: f["recoverable_share"])
    print(f"\n   largest single share: {biggest['id']} at "
          f"{biggest['recoverable_share']*100:.1f}%")

    # ------------------------------------------------------------------ write
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weeks": int(df.week.nunique()),
        "markets": sorted(df.market.unique().tolist()),
        "skus": sorted(df.sku.unique().tolist()),
        "series_fitted": len(base),
        "series_excluded": len(dropped),
        "model": f"SARIMAX{F.ORDER}{F.SEASONAL}",
        "exog": F.EXOG,
        "holdout_weeks": F.TEST_WEEKS,
        "runtime_seconds": round(time.time() - t0, 1),
        "findings": findings,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2, default=_jsonable)

    banner("WRITTEN", "-")
    print(f"   {OUT}   {len(findings)} findings, evidence included")
    print(f"   total runtime {time.time() - t0:.0f}s")
    print("\n   Next:  python store.py   flatten into the tables the app reads")
    print("          python eval.py    the falsifiability layer")
    print("=" * 78 + "\n")


def _wrap(text, width=70):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    main()
