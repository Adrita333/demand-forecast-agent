# -*- coding: utf-8 -*-
"""
generate_data.py - a deliberately small, deliberately flawed demand history.

    python generate_data.py

WHY SYNTHETIC, AND WHY FLAWED ON PURPOSE
The point of this build is NOT "here is a better model". It is "here is why the
existing model looks unstable while behaving correctly". So the data has to
contain the real-world faults that produce that result:

  1  CENSORED DEMAND.  When stock runs out, the record shows what was SHIPPED,
     not what customers wanted. observed_shipments <= true_demand, and the gap
     is invisible to anyone training on shipments. Every retrain learns the
     supply constraint instead of the demand.

  2  EVENTS THE MODEL CANNOT SEE.  Ramadan moves ~11 days earlier each year, so
     it is NOT annual seasonality. A plain SARIMA with a 52-week period cannot
     represent it. It needs to arrive as an exogenous regressor - which is the
     entire argument for SARIMAX over SARIMA.

  3  PROMOTIONS.  Campaign-driven spikes. Without the campaign calendar as a
     regressor they look like outliers, and an analyst "cleans" them out -
     removing the very demand that caused the stockout.

  4  A NEW PRODUCT WITH NO HISTORY.  The diaper launches in week 118. No time
     series method can forecast a series that does not exist yet. Any model
     that appears to is fitting noise.

  5  LEAD TIME LONGER THAN THE FORECAST HORIZON.  Indonesia is 6 weeks; the
     legacy tool forecasts 4. Those orders are placed blind, and no accuracy
     improvement can fix that.

WHAT IS AND IS NOT IN THE FILES
true_demand is the answer key. It exists ONLY so the evaluation can show what
the censored history hides. A real project never has it - which is the point,
and is why the eval stage quarantines it.

    data/demand.csv     week x market x sku, shipments, stock, censoring
    data/events.csv     Ramadan, Chinese New Year, year-end, by market
    data/promos.csv     the campaign calendar
    data/skus.csv       launch week, supplier lead time, price
"""

import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
D = "data"
os.makedirs(D, exist_ok=True)

WEEKS = 156                      # three years, weekly
START = pd.Timestamp("2023-08-28")   # a Monday
dates = [START + pd.Timedelta(weeks=w) for w in range(WEEKS)]

MARKETS = ["SG", "MY", "ID"]
LEAD_TIME = {"SG": 3, "MY": 4, "ID": 6}      # weeks, supplier to warehouse
LEGACY_HORIZON = 4                            # what the existing tool forecasts

SKUS = [
    # sku,        name,                 launch_wk, base by market,       price
    ("PAD-REG",  "Day Pad 24cm",            0, {"SG": 900, "MY": 700, "ID": 1400}, 4.90),
    ("PAD-NGT",  "Night Pad 33cm",          0, {"SG": 520, "MY": 380, "ID": 700},  5.60),
    ("PAD-LIN",  "Panty Liner",             0, {"SG": 340, "MY": 260, "ID": 520},  3.80),
    ("DIA-NB",   "Newborn Diaper",        118, {"SG": 260, "MY": 210, "ID": 480},  8.40),
]

# --------------------------------------------------------------- events
# Ramadan moves about 11 days earlier each Gregorian year. That is exactly why
# it cannot be captured by a fixed 52-week seasonal term.
RAMADAN_START_WK = {2024: 28, 2025: 76, 2026: 124}   # week index of each start
CNY_WK = {2024: 21, 2025: 74, 2026: 125}

ev = []
for w in range(WEEKS):
    d = dates[w]
    ram = any(s <= w < s + 5 for s in RAMADAN_START_WK.values())
    # the two weeks BEFORE Eid are the buying peak, not Ramadan itself
    ram_peak = any(s + 3 <= w < s + 5 for s in RAMADAN_START_WK.values())
    cny = any(c <= w < c + 3 for c in CNY_WK.values())
    yearend = d.month == 12 and d.day >= 8
    for m in MARKETS:
        ev.append({
            "week": w, "week_start": d.date().isoformat(), "market": m,
            "is_ramadan": int(ram and m in ("MY", "ID")),
            "is_ramadan_peak": int(ram_peak and m in ("MY", "ID")),
            "is_cny": int(cny and m in ("SG", "MY")),
            "is_yearend": int(yearend),
        })
events = pd.DataFrame(ev)

# --------------------------------------------------------------- promos
# A real campaign calendar: platform sale days plus brand pushes.
pr = []
for w in range(WEEKS):
    d = dates[w]
    # 9.9 / 10.10 / 11.11 / 12.12 marketplace events
    big = d.month in (9, 10, 11, 12) and 6 <= d.day <= 13
    for m in MARKETS:
        for sku, *_ in SKUS:
            on = big or (rng.random() < 0.06)
            depth = float(rng.choice([0.10, 0.15, 0.20, 0.30])) if on else 0.0
            pr.append({"week": w, "market": m, "sku": sku,
                       "on_promo": int(on), "discount_depth": round(depth, 2)})
promos = pd.DataFrame(pr)

# --------------------------------------------------------------- demand
rows = []
for sku, name, launch, base_by_mkt, price in SKUS:
    for m in MARKETS:
        base = base_by_mkt[m]
        # a slow underlying growth trend, different by market
        growth = {"SG": 0.0012, "MY": 0.0020, "ID": 0.0035}[m]
        on_hand = base * 3.0          # opening stock
        for w in range(WEEKS):
            if w < launch:
                rows.append({
                    "week": w, "week_start": dates[w].date().isoformat(),
                    "market": m, "sku": sku,
                    "true_demand": 0, "observed_shipments": 0,
                    "on_hand_start": 0, "was_censored": 0, "lost_units": 0,
                })
                continue

            weeks_live = w - launch
            level = base * (1 + growth) ** weeks_live
            # a new product ramps rather than starting at full rate
            if weeks_live < 12:
                level *= 0.45 + 0.055 * weeks_live

            # annual seasonality - a genuine 52-week cycle
            seas = 1 + 0.10 * np.sin(2 * np.pi * (w % 52) / 52 - 0.6)

            e = events[(events.week == w) & (events.market == m)].iloc[0]
            mult = 1.0
            if e.is_ramadan:
                mult *= 1.15
            if e.is_ramadan_peak:
                mult *= 1.55          # the pre-Eid stock-up
            if e.is_cny:
                mult *= 1.30
            if e.is_yearend:
                mult *= 1.12

            p = promos[(promos.week == w) & (promos.market == m)
                       & (promos.sku == sku)].iloc[0]
            if p.on_promo:
                mult *= 1 + 2.2 * p.discount_depth   # promo lift

            noise = rng.normal(1.0, 0.11)
            true_demand = max(0, int(level * seas * mult * noise))

            # --- replenishment, deliberately naive -------------------------
            # Orders are placed on a 4-week moving average of what was SHIPPED,
            # with no safety stock and no service-level target. This is the
            # policy that produces the stockouts - not the model.
            hist = [r["observed_shipments"] for r in rows[-8:]
                    if r["market"] == m and r["sku"] == sku]
            reorder = np.mean(hist[-4:]) if len(hist) >= 4 else level
            arrival = reorder * 1.0            # what lands this week

            available = on_hand + arrival
            shipped = int(min(true_demand, available))
            censored = int(shipped < true_demand)
            lost = true_demand - shipped
            on_hand = max(0.0, available - shipped)

            rows.append({
                "week": w, "week_start": dates[w].date().isoformat(),
                "market": m, "sku": sku,
                "true_demand": true_demand,
                "observed_shipments": shipped,
                "on_hand_start": int(available),
                "was_censored": censored,
                "lost_units": int(lost),
            })

demand = pd.DataFrame(rows)

sku_tbl = pd.DataFrame([
    {"sku": s, "name": n, "launch_week": lw,
     "unit_price_usd": p,
     "lead_time_weeks_SG": LEAD_TIME["SG"],
     "lead_time_weeks_MY": LEAD_TIME["MY"],
     "lead_time_weeks_ID": LEAD_TIME["ID"]}
    for s, n, lw, _, p in SKUS])

demand.to_csv(f"{D}/demand.csv", index=False)
events.to_csv(f"{D}/events.csv", index=False)
promos.to_csv(f"{D}/promos.csv", index=False)
sku_tbl.to_csv(f"{D}/skus.csv", index=False)

# --------------------------------------------------------------- report
live = demand[demand.true_demand > 0]
n = len(live)
cens = live[live.was_censored == 1]
price = {s: p for s, _, _, _, p in SKUS}
lost_usd = (live.lost_units * live.sku.map(price)).sum()

print("\n" + "=" * 74)
print(f"  DEMAND HISTORY  ·  {WEEKS} weeks  ·  {len(MARKETS)} markets  "
      f"·  {len(SKUS)} SKUs")
print("=" * 74)
print(f"   {len(demand)} rows written, {n} of them after launch")
print(f"   period: {dates[0].date()} to {dates[-1].date()}")

print("\n  THE FAULT THAT MATTERS  ·  censored demand")
print("-" * 74)
print(f"   {len(cens)} of {n} live weeks ({len(cens)/n*100:.1f}%) stocked out")
print(f"   units customers wanted and did not get   {live.lost_units.sum():,}")
print(f"   at list price                            US${lost_usd:,.0f}")
print("\n   In those weeks observed_shipments is LOWER than true_demand. A model")
print("   trained on shipments learns the supply constraint, forecasts low,")
print("   triggers a smaller order, and stocks out again. The model is behaving")
print("   correctly. The training data is wrong.")

print("\n  WHERE THE STOCKOUTS CLUSTER")
print("-" * 74)
lv = live.merge(events, on=["week", "market"], how="left")
for lbl, mask in (("Ramadan peak weeks", lv.is_ramadan_peak == 1),
                  ("Chinese New Year", lv.is_cny == 1),
                  ("promo weeks", lv.merge(promos, on=["week", "market", "sku"],
                                           how="left").on_promo == 1),
                  ("ordinary weeks", (lv.is_ramadan_peak == 0) & (lv.is_cny == 0))):
    sub = lv[mask]
    if len(sub):
        print(f"   {lbl:<22}{len(sub):>6} weeks   "
              f"{sub.was_censored.mean()*100:>5.1f}% stocked out")
print("\n   They cluster on events, not at random. Ramadan moves ~11 days earlier")
print("   every year, so a 52-week seasonal term cannot represent it. That is the")
print("   whole argument for SARIMAX with the event calendar as a regressor.")

print("\n  BY MARKET")
print("-" * 74)
print(f"   {'market':<10}{'lead time':>11}{'stockout %':>13}{'lost US$':>14}")
for m in MARKETS:
    sub = live[live.market == m]
    usd = (sub.lost_units * sub.sku.map(price)).sum()
    print(f"   {m:<10}{LEAD_TIME[m]:>9} wk{sub.was_censored.mean()*100:>12.1f}%"
          f"{usd:>14,.0f}")
print(f"\n   The legacy tool forecasts {LEGACY_HORIZON} weeks ahead. Indonesia's lead time")
print(f"   is {LEAD_TIME['ID']} weeks, so every ID order is placed on a forecast that does")
print("   not reach far enough. No accuracy improvement can fix that - it is an")
print("   operational mismatch, and it is free to correct.")

print("\n  THE NEW PRODUCT")
print("-" * 74)
dia = demand[(demand.sku == "DIA-NB") & (demand.true_demand > 0)]
print(f"   DIA-NB launches in week 118 - {len(dia)//3} weeks of history per market.")
print("   No time series method can forecast a series that did not exist. Any")
print("   model that appears to is fitting noise, and saying so is the answer.")

print("\n" + "-" * 74)
print("  FILES")
print("-" * 74)
for f in ("demand", "events", "promos", "skus"):
    df = pd.read_csv(f"{D}/{f}.csv")
    print(f"   data/{f}.csv{'':<6}{len(df):>6} rows   {', '.join(df.columns[:6])}")
print("\n   true_demand and lost_units are the ANSWER KEY. Nothing that fits a")
print("   model may read them - only the evaluation stage, at the end.")
print("=" * 74 + "\n")
