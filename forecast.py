# -*- coding: utf-8 -*-
"""
forecast.py - the shared modelling layer. Nothing in this file makes a
judgement about the client's process; it only fits, forecasts, backtests and
simulates. The judgements live in checks.py.

    imported by checks.py, main.py and eval.py - not run directly

WHY THIS FILE EXISTS SEPARATELY
Four diagnostics all need the same three things: a SARIMAX fit, a rolling
backtest, and an inventory simulator. If each check carried its own copy they
would drift apart, and a comparison between two checks would stop meaning
anything. One implementation, used by all four, is what makes the findings
comparable.

THE MODEL, AND WHY THIS ONE
    SARIMAX(1,0,1)(0,1,0,52) on log(observed_shipments), with the event
    calendar as exogenous regressors.

  log      variance rises with the level, so the log makes the seasonal swing
           additive instead of multiplicative
  (1,0,1)  ACF and PACF both tail off rather than cutting off, which is the
           signature of a mixed ARMA term
  (0,1,0,52)  one seasonal difference. D=1 removes the annual cycle; no
           seasonal AR or MA term survived on 156 weeks - three cycles is very
           little to estimate them from, and saying so is more honest than
           fitting them anyway
  exog     Ramadan moves ~11 days earlier each year, so it is NOT a 52-week
           seasonal effect and CANNOT be captured by the seasonal term. It has
           to enter as a regressor. That is the entire SARIMA -> SARIMAX step,
           and it is the one modelling choice here that is not cosmetic

THE ANSWER KEY
Every function in this file fits on `observed_shipments`. `true_demand` is
passed in only to `simulate()`, which is an evaluation harness, never a
fitting routine. Nothing here trains on the answer key.
"""

import warnings
from collections import deque

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ config
EXOG = ["is_ramadan_peak", "is_cny", "is_yearend", "on_promo"]
ORDER = (1, 0, 1)
SEASONAL = (0, 1, 0, 52)

TEST_WEEKS = 26            # holdout, never seen during fitting
REVIEW = 1                 # weekly review period
MIN_HISTORY = 80           # a series shorter than this is not fitted at all
LEGACY_HORIZON = 4         # what the incumbent tool produces
LEAD = {"SG": 3, "MY": 4, "ID": 6}


# ------------------------------------------------------------------ loading
def load(data_dir="data"):
    """Join the four tables into one long frame, plus a price lookup."""
    dem = pd.read_csv(f"{data_dir}/demand.csv")
    ev = pd.read_csv(f"{data_dir}/events.csv")
    pro = pd.read_csv(f"{data_dir}/promos.csv")[["week", "market", "sku", "on_promo"]]
    sku = pd.read_csv(f"{data_dir}/skus.csv")
    df = dem.merge(ev.drop(columns=["week_start"]), on=["week", "market"])
    df = df.merge(pro, on=["week", "market", "sku"])
    price = dict(zip(sku.sku, sku.unit_price_usd))
    launch = dict(zip(sku.sku, sku.launch_week))
    return df, price, launch


def series_list(df):
    """
    Every market x SKU pair with enough history to fit.

    A series shorter than MIN_HISTORY is EXCLUDED, not fitted badly. The new
    product has 38 weeks against a 52-week seasonal period - the seasonal term
    cannot be estimated from less than one full cycle, and a model that appears
    to forecast it is fitting noise. Excluding it is the finding, not a gap.
    """
    keep, dropped = [], []
    for (m, s), g in df.groupby(["market", "sku"]):
        g = g.sort_values("week").reset_index(drop=True)
        g = g[g.observed_shipments > 0].reset_index(drop=True)
        if len(g) >= MIN_HISTORY:
            keep.append((m, s, g))
        else:
            dropped.append((m, s, len(g)))
    return keep, dropped


# ------------------------------------------------------------------ fitting
def fit(y, X):
    """One SARIMAX fit. Returns the fitted results object, or None."""
    try:
        return SARIMAX(y, exog=X, order=ORDER, seasonal_order=SEASONAL,
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False)
    except Exception:
        return None


def fit_forecast(g, cut, steps, y_override=None):
    """
    Fit on g[:cut], forecast `steps` weeks.

    y_override lets a caller substitute a repaired target - that is how the
    censoring check compares an EM-imputed history against the raw one without
    duplicating any of this machinery.

    Returns (forecast, actual_shipped, true_demand, sigma) or four Nones.
    sigma is the standard deviation of the in-sample one-step error, which is
    what an order-up-to policy needs to size a buffer.
    """
    tr, te = g.iloc[:cut], g.iloc[cut:cut + steps]
    if len(te) < steps:
        return None, None, None, None

    y = (np.log(tr.observed_shipments.astype(float).clip(lower=1))
         if y_override is None else y_override)
    res = fit(y, tr[EXOG].astype(float))
    if res is None:
        return None, None, None, None

    try:
        f = np.exp(res.forecast(steps=steps, exog=te[EXOG].astype(float)).values)
    except Exception:
        return None, None, None, None

    insample = np.exp(res.fittedvalues.values)
    err = tr.observed_shipments.values[-len(insample):] - insample
    err = err[np.isfinite(err)]
    sigma = float(np.std(err)) if len(err) else float(np.std(tr.observed_shipments))

    return (f,
            te.observed_shipments.values.astype(float),
            te.true_demand.values.astype(float),
            sigma)


def rolling_origin(g, origins, horizon):
    """
    Refit at each origin and record the error AT EACH HORIZON separately.

    Blending horizons into one average is the usual mistake: it hides the fact
    that a team reporting one-step accuracy is reporting accuracy at a horizon
    that drives no decision. Returns {h: [abs pct errors]}.
    """
    out = {h: [] for h in range(1, horizon + 1)}
    for cut in origins:
        if cut < MIN_HISTORY:
            continue
        f, a, _, _ = fit_forecast(g, cut, horizon)
        if f is None:
            continue
        for h in range(1, horizon + 1):
            out[h].append(abs(f[h - 1] - a[h - 1]) / max(a[h - 1], 1.0))
    return out


def em_impute(g, cut, rounds=3):
    """
    Censored-likelihood imputation, EM style.

    In a stockout week the record is a LOWER BOUND on demand, not demand. The
    fix is not to delete those weeks - they cluster on peaks, so deleting them
    removes the high observations and biases the forecast further down. The fix
    is to treat them as censored: fit, predict what demand would have been, and
    replace the observation with max(observed, predicted). Never below what was
    actually shipped, because that much definitely sold.

    Returns the repaired log-target for g[:cut], or None.
    """
    tr = g.iloc[:cut]
    y0 = np.log(tr.observed_shipments.astype(float).clip(lower=1)).reset_index(drop=True)
    mask = (tr.was_censored == 1).values
    if not mask.any():
        return y0
    X = tr[EXOG].astype(float).reset_index(drop=True)

    y = y0.copy()
    for _ in range(rounds):
        res = fit(y, X)
        if res is None:
            return y0
        pred = pd.Series(res.fittedvalues).reset_index(drop=True)
        y = y.copy()
        y[mask] = np.maximum(y0[mask].values, pred[mask].values)
    return y


# ------------------------------------------------------------------ metrics
def mape(a, f):
    m = a > 0
    return float(np.mean(np.abs(f[m] - a[m]) / a[m]) * 100)


def wmape(a, f):
    return float(np.sum(np.abs(f - a)) / np.sum(a) * 100)


def bias(a, f):
    return float((np.mean(f) - np.mean(a)) / np.mean(a) * 100)


# ------------------------------------------------------------------ policy
def simulate(true_d, fcst, sigma, lead, z, horizon=None):
    """
    Weekly periodic review, order-up-to S, fixed lead time.

        S = sum(forecast over lead + review) + z * sigma * sqrt(lead + review)

    `horizon` caps how far ahead the planner can SEE. Beyond it the forecast
    does not exist, so the gap is filled the way a human fills it - flat-line
    the average of the weeks they do have. No seasonality, no events. Pass None
    for full visibility.

    TWO WARM-START RULES, both of which were wrong in an earlier draft and both
    of which flattered the result:

      1  the inventory POSITION (on hand + in transit) starts at exactly S, not
         above it. Start it higher and the first weeks coast on a cushion the
         policy would never have built.
      2  the safety stock is PRESENT at week zero, not accumulated over the
         first lead-time weeks. Otherwise every z behaves identically until the
         first policy-driven order lands, and losses in those weeks get blamed
         on a buffer that had not arrived yet.

    Returns (lost_units, demanded_units, stockout_weeks, mean_on_hand).
    """
    P = lead + REVIEW
    base = float(np.mean(fcst))
    on_hand = base * REVIEW + z * sigma * np.sqrt(P)
    pipeline = deque([base] * lead)

    lost = demanded = short_weeks = 0.0
    held = []

    for t in range(len(true_d)):
        on_hand += pipeline.popleft() if pipeline else 0.0
        d = float(true_d[t])
        shipped = min(on_hand, d)
        on_hand -= shipped
        lost += d - shipped
        demanded += d
        short_weeks += 1 if shipped < d - 1e-9 else 0
        held.append(on_hand)

        vis = fcst[t + 1: t + 1 + (P if horizon is None else min(horizon, P))]
        if len(vis) == 0:
            vis = np.array([fcst[-1]])
        if len(vis) < P:
            vis = np.concatenate([vis, np.repeat(np.mean(vis), P - len(vis))])
        S = float(np.sum(vis)) + z * sigma * np.sqrt(P)
        pipeline.append(max(0.0, S - (on_hand + sum(pipeline))))

    return lost, demanded, short_weeks, float(np.mean(held))
