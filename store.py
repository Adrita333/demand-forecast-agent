# -*- coding: utf-8 -*-
"""
store.py - flatten store/findings.json into the tables the app reads.

    python store.py        (run main.py first)

WHY A SEPARATE STAGE
The app must compute nothing at display time. A dashboard that refits a model
when someone moves a filter is a dashboard that will fail in front of a client,
and worse, it lets the displayed number drift from the number that was scored.
So the pipeline writes, and the app reads. What a reviewer sees is exactly what
was measured, and it is reproducible from the JSON.

WHAT IT WRITES
    store/findings.csv          one row per gate - the report itself
    store/scorecard.csv         the headline numbers, one row
    store/metric_sweep.csv      MAPE / WMAPE / bias against the scaling k
    store/service_curve.csv     fill rate and cost against the service level z
    store/horizon_curve.csv     forecast error at each horizon separately
    store/horizon_markets.csv   where the horizon loss actually sits
    store/horizon_zsens.csv     how much of it a buffer absorbs
    store/excluded_series.csv   series too young to fit

Every one of these comes from findings.json. Nothing here recomputes anything,
which means store.py cannot silently disagree with main.py.
"""

import json
import os

import pandas as pd

STORE = "store"
SRC = f"{STORE}/findings.json"


def _get(findings, gate_id):
    for f in findings:
        if f["id"] == gate_id:
            return f
    return None


def _write(df, name):
    path = f"{STORE}/{name}"
    df.to_csv(path, index=False)
    print(f"   {path:<34}{len(df):>5} rows")
    return path


def main():
    if not os.path.exists(SRC):
        raise SystemExit(
            f"{SRC} not found. Run  python main.py  first - it is the stage "
            "that fits the models and produces the findings.")

    with open(SRC) as fh:
        d = json.load(fh)
    F = d["findings"]

    print("\n" + "=" * 74)
    print("  STORE  ·  flattening findings into the tables the app reads")
    print("=" * 74)

    # ------------------------------------------------------------ findings
    rows = []
    for i, f in enumerate(F, 1):
        rows.append({
            "rank": i,
            "gate": f["id"],
            "title": f["title"],
            "severity": f["severity"],
            "recoverable_share_pct": round(f["recoverable_share"] * 100, 2),
            "cost_to_fix": f["cost_to_fix"],
            "diagnosis": f["diagnosis"],
            "validation": f["validation"],
            "caveat": f["caveat"],
        })
    _write(pd.DataFrame(rows), "findings.csv")

    # ------------------------------------------------------------ scorecard
    g1, g2 = _get(F, "G1-METRIC"), _get(F, "G2-SAFETY-STOCK")
    g3, g4 = _get(F, "G3-CENSORED-HISTORY"), _get(F, "G4-HORIZON")
    g5 = _get(F, "G5-COVERAGE")

    biggest = max(F, key=lambda f: f["recoverable_share"])
    card = {
        "generated_utc": d["generated_utc"],
        "weeks": d["weeks"],
        "markets": len(d["markets"]),
        "skus": len(d["skus"]),
        "series_fitted": d["series_fitted"],
        "series_excluded": d["series_excluded"],
        "model": d["model"],
        "holdout_weeks": d["holdout_weeks"],
        "runtime_seconds": d["runtime_seconds"],
        "gates_run": len(F),
        "gates_high": sum(1 for f in F if f["severity"] == "HIGH"),
        "gates_clear": sum(1 for f in F if f["severity"] == "CLEAR"),
        "largest_gate": biggest["id"],
        "largest_share_pct": round(biggest["recoverable_share"] * 100, 2),
        # the two numbers a planner would recognise on sight
        "censoring_rate_pct": g3["evidence"]["censoring_rate_pct"] if g3 else None,
        "bias_naive_pct": g3["evidence"]["bias_naive_pct"] if g3 else None,
        "bias_repaired_pct": g3["evidence"]["bias_em_pct"] if g3 else None,
        "k_mape": g1["evidence"]["k_mape"] if g1 else None,
        "k_wmape": g1["evidence"]["k_wmape"] if g1 else None,
        "z_recommended": g2["evidence"]["z_recommended"] if g2 else None,
        "net_usd_safety_stock": g2["evidence"]["net_usd"] if g2 else None,
    }
    _write(pd.DataFrame([card]), "scorecard.csv")

    # ------------------------------------------------------------ evidence
    if g1:
        _write(pd.DataFrame(g1["evidence"]["sweep"]), "metric_sweep.csv")
    if g2:
        _write(pd.DataFrame(g2["evidence"]["curve"]), "service_curve.csv")
    if g4:
        _write(pd.DataFrame(g4["evidence"]["error_by_horizon"]),
               "horizon_curve.csv")
        # by_market already carries lead time and required horizon per market,
        # so the separate gaps list would be a duplicate. One table, not two.
        _write(pd.DataFrame(g4["evidence"]["by_market"]), "horizon_markets.csv")
        _write(pd.DataFrame(g4["evidence"]["z_sensitivity"]),
               "horizon_zsens.csv")
    if g5:
        ex = pd.DataFrame(g5["evidence"]["excluded"])
        if ex.empty:
            ex = pd.DataFrame(columns=["market", "sku", "weeks"])
        _write(ex, "excluded_series.csv")

    # ------------------------------------------------------------ report
    print("\n" + "-" * 74)
    print("  WHAT A REVIEWER SEES")
    print("-" * 74)
    print(f"   {'':<3}{'gate':<22}{'severity':<10}{'recovers':>10}")
    for i, f in enumerate(F, 1):
        print(f"   {i:<3}{f['id']:<22}{f['severity']:<10}"
              f"{f['recoverable_share']*100:>9.1f}%")
    print(f"\n   {card['gates_high']} of {card['gates_run']} gates fired HIGH. "
          f"{card['gates_clear']} came back CLEAR.")
    print("   A gate that finds nothing still appears in the report. A")
    print("   diagnostic that can only ever confirm is not a diagnostic.")

    print("\n" + "-" * 74)
    print("  THE HEADLINE, STATED HONESTLY")
    print("-" * 74)
    print(f"   Largest single recoverable share: {card['largest_gate']} at "
          f"{card['largest_share_pct']:.1f}%.")
    print("   The shares OVERLAP and are not additive - censored data and the")
    print("   metric both push the forecast down, and a buffer absorbs most of")
    print("   the horizon shortfall. Quote the largest, note that the others")
    print("   compound it, and do not sum them.")

    print("\n" + "=" * 74)
    print("   Next:  python eval.py            the falsifiability layer")
    print("          python -m streamlit run app.py")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
