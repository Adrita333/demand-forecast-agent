# -*- coding: utf-8 -*-
"""
checks.py - the diagnostic gates. This is the agent.

    imported by main.py and eval.py - not run directly

WHAT THIS FILE IS FOR
A demand forecast that keeps stocking out is almost never fixed by a bigger
model. It is fixed by finding which of a small number of well-understood
faults is actually present. Each gate below tests for one, on the client's own
history, and returns a finding in a fixed shape:

    id                 short code
    title              one line a planner would recognise
    severity           HIGH / MEDIUM / LOW / CLEAR
    diagnosis          what is wrong, in plain words
    evidence           the numbers, computed not asserted
    recoverable_share  fraction of stockout loss this fix returns
    cost_to_fix        engineering effort, honestly stated
    validation         a test the CLIENT can run that could prove this wrong
    caveat             what this finding does not establish

A gate that finds nothing returns severity CLEAR and stays in the report. A
diagnostic that can only ever confirm is not a diagnostic.

THE ONE RULE
No gate reads true_demand. Gates fit and forecast on observed_shipments only.
true_demand enters exclusively through simulate(), which scores a policy after
the fact - the evaluation stage, never the fitting stage.
"""

import numpy as np
import pandas as pd

import forecast as F

# a fix is only worth reporting if it moves something by at least this much
MATERIAL = 0.02

# what money costs to sit in a warehouse for a year
CARRYING_RATE = 0.22

# the k sweep for the metric gate
KS = np.round(np.arange(0.70, 1.31, 0.05), 2)

# service-level multipliers, and the fill rate each nominally targets
Z_LEVELS = [(0.00, "no buffer"), (0.84, "80%"), (1.28, "90%"),
            (1.65, "95%"), (2.05, "98%")]


def _finding(**kw):
    """Every gate returns this shape, so the store layer stays trivial."""
    base = {"id": "", "title": "", "severity": "CLEAR", "diagnosis": "",
            "evidence": {}, "recoverable_share": 0.0, "cost_to_fix": "",
            "validation": "", "caveat": ""}
    base.update(kw)
    return base


# ---------------------------------------------------------------- base fits
def base_forecasts(series):
    """
    One holdout forecast per series, fitted once and shared by three gates.

    Refitting inside each gate would be slower and, worse, would let the gates
    disagree with each other about the same series. One fit, reused.
    """
    out = []
    for m, s, g in series:
        cut = len(g) - F.TEST_WEEKS
        f, a, td, sigma = F.fit_forecast(g, cut, F.TEST_WEEKS)
        if f is None:
            continue
        out.append({"market": m, "sku": s, "series": g, "cut": cut,
                    "forecast": f, "actual": a, "true_demand": td,
                    "sigma": sigma})
    return out


def _usd(bf, units):
    return units * bf["price"]


# ---------------------------------------------------------------- gate 1
def check_metric(base, price):
    """
    GATE 1 - is the accuracy metric asking for the behaviour that empties the
    shelf?

    MAPE = mean(|forecast - actual| / actual). The denominator is the actual,
    so under-forecasting to zero scores at worst 100%, while over-forecasting
    by 5x scores 400%. The punishment for being low has a floor; the punishment
    for being high does not. The cheapest way to minimise MAPE is to forecast
    low, every time.

    THE TEST. Take one forecast, multiply by k, sweep k, and score each version
    on MAPE, WMAPE and bias. Nothing changes but the ruler. If MAPE were
    symmetric its minimum would sit at k = 1.00. Where it actually sits is the
    finding.
    """
    a = np.concatenate([b["actual"] for b in base])
    f0 = np.concatenate([b["forecast"] for b in base])

    rows = [{"k": float(k), "mape": F.mape(a, f0 * k),
             "wmape": F.wmape(a, f0 * k), "bias": F.bias(a, f0 * k)}
            for k in KS]
    R = pd.DataFrame(rows)
    k_mape = float(R.loc[R.mape.idxmin(), "k"])
    k_wmape = float(R.loc[R.wmape.idxmin(), "k"])
    bias_mape = float(R.loc[R.k == k_mape, "bias"].iloc[0])
    bias_wmape = float(R.loc[R.k == k_wmape, "bias"].iloc[0])

    # price it: order to each version under an identical policy
    def lost_usd(k):
        tot = 0.0
        for b in base:
            lost, _, _, _ = F.simulate(b["true_demand"], b["forecast"] * k,
                                       b["sigma"], F.LEAD[b["market"]], z=0.0)
            tot += lost * price[b["sku"]]
        return tot

    d_mape, d_wmape = lost_usd(k_mape), lost_usd(k_wmape)
    share = (d_mape - d_wmape) / d_mape if d_mape > 0 else 0.0

    asymmetric = k_mape < k_wmape - 1e-9
    sev = "HIGH" if asymmetric and share >= MATERIAL else (
        "LOW" if asymmetric else "CLEAR")

    return _finding(
        id="G1-METRIC",
        title="The accuracy metric rewards under-forecasting",
        severity=sev,
        diagnosis=(
            f"MAPE is minimised at k={k_mape:.2f} (bias {bias_mape:+.1f}%) while "
            f"WMAPE is minimised at k={k_wmape:.2f} (bias {bias_wmape:+.1f}%). "
            "Same model, same data, same holdout - only the ruler changed, and "
            "the ruler asks for a smaller number. A team that tunes to MAPE is "
            "being told to under-forecast, and it will."
            if asymmetric else
            "On this history MAPE and WMAPE select the same scaling. The "
            "asymmetry is real in the formula but is not material here."),
        evidence={
            "k_mape": k_mape, "k_wmape": k_wmape,
            "bias_at_k_mape_pct": round(bias_mape, 2),
            "bias_at_k_wmape_pct": round(bias_wmape, 2),
            "bias_unscaled_pct": round(float(R.loc[R.k == 1.00, "bias"].iloc[0]), 2),
            "mape_at_k_mape": round(float(R.mape.min()), 2),
            "wmape_at_k_wmape": round(float(R.wmape.min()), 2),
            "usd_lost_tuned_to_mape": round(d_mape),
            "usd_lost_tuned_to_wmape": round(d_wmape),
            "sweep": R.round(2).to_dict("records"),
        },
        recoverable_share=share,
        cost_to_fix="One line in the scoring script. No new model, data or system.",
        validation=(
            "Re-score last year's forecasts on WMAPE and bias alongside MAPE. "
            "If the model the team selected is not the one WMAPE would have "
            "selected, the metric is choosing for them. This needs no new data "
            "and no true-demand column."),
        caveat=(
            "The SIZE of the effect is a property of this error distribution. "
            "The DIRECTION is a property of the formula and holds anywhere. "
            "MAPE is not useless - it is scale-free and easy to explain, which "
            "is why it spread. It is the wrong choice for an inventory "
            "decision, which is a different claim."),
    )


# ---------------------------------------------------------------- gate 2
def check_safety_stock(base, price):
    """
    GATE 2 - is the order sized on the forecast alone?

    Ordering to a point forecast means aiming at the middle of the error
    distribution, which stocks out roughly half the time by construction. A
    service level is a decision about how much of the error to absorb:

        S = sum(forecast over lead + review) + z * sigma * sqrt(lead + review)

    z is the only lever, and it is a business choice, not a modelling one. This
    gate prices the whole curve so the client can pick a point rather than be
    handed one.
    """
    rows = []
    for z, label in Z_LEVELS:
        lost_u = dem_u = short_w = weeks = 0.0
        held = []
        usd = 0.0
        for b in base:
            l, d, sw, oh = F.simulate(b["true_demand"], b["forecast"],
                                      b["sigma"], F.LEAD[b["market"]], z=z)
            lost_u += l
            dem_u += d
            short_w += sw
            weeks += len(b["true_demand"])
            held.append(oh)
            usd += l * price[b["sku"]]
        rows.append({"z": z, "target": label,
                     "fill_rate_pct": (1 - lost_u / dem_u) * 100,
                     "weeks_ok_pct": (1 - short_w / weeks) * 100,
                     "usd_lost": usd,
                     "mean_units_held": float(np.mean(held))})
    R = pd.DataFrame(rows)

    base_row = R.iloc[0]                       # z = 0, today's policy
    # the cheapest z that gets fill rate to 99.5% or better, else the best one
    ok = R[R.fill_rate_pct >= 99.5]
    pick = ok.iloc[0] if len(ok) else R.iloc[-1]

    # what the extra stock costs to hold
    unit_price = float(np.mean([price[b["sku"]] for b in base]))
    extra_units = float(pick.mean_units_held - base_row.mean_units_held) * len(base)
    extra_value = extra_units * unit_price
    carrying = extra_value * CARRYING_RATE
    gross = float(base_row.usd_lost - pick.usd_lost)
    net = gross - carrying
    share = gross / base_row.usd_lost if base_row.usd_lost > 0 else 0.0

    sev = "HIGH" if share >= 0.10 else ("MEDIUM" if share >= MATERIAL else "CLEAR")

    return _finding(
        id="G2-SAFETY-STOCK",
        title="Orders are sized on the point forecast, with no buffer",
        severity=sev,
        diagnosis=(
            f"At z=0 the simulated fill rate is {base_row.fill_rate_pct:.1f}% and "
            f"{100 - base_row.weeks_ok_pct:.1f}% of weeks stock out. Raising z to "
            f"{pick.z:.2f} - a {pick.target} service target - lifts fill to "
            f"{pick.fill_rate_pct:.1f}%. The forecast is unchanged. What changes is "
            "how much of its error the policy is willing to absorb."),
        evidence={
            "curve": R.round(2).to_dict("records"),
            "z_recommended": float(pick.z),
            "gross_usd_recovered": round(gross),
            "extra_stock_value_usd": round(extra_value),
            "carrying_cost_usd": round(carrying),
            "net_usd": round(net),
            "carrying_rate": CARRYING_RATE,
        },
        recoverable_share=share,
        cost_to_fix=("A parameter in the replenishment rule, plus a business "
                     "decision on the service target. No model change."),
        validation=(
            "Compare the realised fill rate against the service level the "
            "policy nominally targets. If the policy has no z at all, the "
            "target is 50% whether anyone chose it or not. That comparison "
            "runs on their own shipment and order history."),
        caveat=(
            "THIS SHARE IS AN UPPER BOUND. The simulation has fixed lead times, "
            "no supplier shortfalls, no minimum order quantities and no shelf "
            "constraint. Every one of those pushes the real recovery lower. "
            "Quote the direction and the mechanism; treat the level as a "
            "ceiling, not an estimate."),
    )


# ---------------------------------------------------------------- gate 3
def check_censoring(series, price, base):
    """
    GATE 3 - is the training data recording demand, or recording supply?

    In a stockout week the system records what was SHIPPED. That is a lower
    bound on demand, not demand. Train on it and the model learns the supply
    constraint: it forecasts low, triggers a smaller order, and stocks out
    again. The model is behaving correctly. The data is wrong.

    The obvious fix - drop the stockout weeks - MAKES IT WORSE, and this gate
    is built to show that. Stockouts cluster on peaks, so dropping them removes
    the high observations. The correct treatment is censored-likelihood
    imputation: replace the observation with max(shipped, predicted).
    """
    n_c = n = 0
    for _, _, g in series:
        n_c += int(g.was_censored.sum())
        n += len(g)
    cens_rate = n_c / n if n else 0.0

    naive_bias, em_bias = [], []
    usd_naive = usd_em = 0.0
    repaired = 0

    for b in base:
        g, cut = b["series"], b["cut"]
        naive_bias.append(F.bias(b["actual"], b["forecast"]))

        y_em = F.em_impute(g, cut)
        f_em, a_em, td_em, sig_em = F.fit_forecast(g, cut, F.TEST_WEEKS,
                                                   y_override=y_em)
        if f_em is None:
            f_em, sig_em = b["forecast"], b["sigma"]
        else:
            repaired += 1
        em_bias.append(F.bias(b["actual"], f_em))

        l_n, _, _, _ = F.simulate(b["true_demand"], b["forecast"],
                                  b["sigma"], F.LEAD[b["market"]], z=0.0)
        l_e, _, _, _ = F.simulate(b["true_demand"], f_em,
                                  sig_em, F.LEAD[b["market"]], z=0.0)
        usd_naive += l_n * price[b["sku"]]
        usd_em += l_e * price[b["sku"]]

    nb, eb = float(np.mean(naive_bias)), float(np.mean(em_bias))
    share = (usd_naive - usd_em) / usd_naive if usd_naive > 0 else 0.0
    sev = "HIGH" if cens_rate >= 0.05 and share >= MATERIAL else (
        "MEDIUM" if cens_rate >= 0.05 else "CLEAR")

    return _finding(
        id="G3-CENSORED-HISTORY",
        title="The model is trained on shipments, not on demand",
        severity=sev,
        diagnosis=(
            f"{cens_rate*100:.1f}% of live weeks stocked out. In those weeks the "
            f"record is a floor on demand, not demand. Forecast bias on raw "
            f"shipments is {nb:+.1f}%; after censored-likelihood imputation it is "
            f"{eb:+.1f}%. The gap closed is the part of the low bias that the "
            "data caused rather than the model."),
        evidence={
            "censored_weeks": n_c, "live_weeks": n,
            "censoring_rate_pct": round(cens_rate * 100, 2),
            "bias_naive_pct": round(nb, 2),
            "bias_em_pct": round(eb, 2),
            "bias_points_recovered": round(nb - eb, 2),
            "series_repaired": repaired,
            "usd_lost_naive": round(usd_naive),
            "usd_lost_em": round(usd_em),
        },
        recoverable_share=share,
        cost_to_fix=("A stockout flag on the demand table, then an EM loop "
                     "around the existing fit. Days, not weeks - but it needs "
                     "the flag, and most systems do not store one."),
        validation=(
            "THE PLACEBO. Take weeks that did NOT stock out, artificially "
            "censor them at a level you choose, refit, and measure how far the "
            "bias moves. You know the truth for those weeks because you "
            "withheld it yourself. If the repair recovers most of the bias you "
            "induced, the method works - and this needs no true-demand column, "
            "which is why it runs on real client data."),
        caveat=(
            "Deleting or masking the stockout weeks is the intuitive fix and "
            "it makes the bias WORSE, because stockouts cluster on peak weeks "
            "and masking removes the high observations. That result is "
            "reproduced in eval.py rather than asserted here. Imputation also "
            "cannot recover demand that never appeared as an order - customers "
            "who saw an empty shelf and bought nothing leave no trace at all."),
    )


# ---------------------------------------------------------------- gate 4
def check_horizon(series, base, price, legacy_horizon=F.LEGACY_HORIZON):
    """
    GATE 4 - does the forecast reach the week the order is responsible for?

    An order placed today lands in L weeks and must cover demand until the next
    order lands. The weeks it is accountable for are L to L+R. So:

        required horizon = lead time + review period

    If the tool forecasts fewer weeks than that, the planner fills the gap by
    hand and that guess sizes the order. No accuracy improvement can fix it,
    and correcting it costs nothing.
    """
    gaps = {}
    for m, lead in F.LEAD.items():
        need = lead + F.REVIEW
        gaps[m] = {"lead": lead, "required": need,
                   "provided": legacy_horizon, "short": max(0, need - legacy_horizon)}
    worst = max(g["short"] for g in gaps.values())

    # how error grows with horizon - measured, at each horizon separately
    H = max(8, max(g["required"] for g in gaps.values()))
    err = {h: [] for h in range(1, H + 1)}
    for m, s, g in series:
        n = len(g)
        for cut in [n - 26, n - 20, n - 14, n - 8]:
            r = F.rolling_origin(g, [cut], H)
            for h, v in r.items():
                err[h].extend(v)
    curve = [{"horizon": h, "wmape_pct": round(float(np.mean(v)) * 100, 2),
              "n": len(v)} for h, v in err.items() if v]

    # WHAT THE SHORTFALL COSTS. Scored at z=0, the same bare policy gates 1 and
    # 3 use, so all three measure one variable against a common baseline.
    by_market = {}
    usd_legacy = usd_fixed = 0.0
    for b in base:
        m = b["market"]
        need = gaps[m]["required"]
        l_a, _, _, _ = F.simulate(b["true_demand"], b["forecast"], b["sigma"],
                                  F.LEAD[m], z=0.0, horizon=legacy_horizon)
        l_b, _, _, _ = F.simulate(b["true_demand"], b["forecast"], b["sigma"],
                                  F.LEAD[m], z=0.0, horizon=need)
        a_usd, b_usd = l_a * price[b["sku"]], l_b * price[b["sku"]]
        usd_legacy += a_usd
        usd_fixed += b_usd
        d = by_market.setdefault(m, {"market": m, "lead": F.LEAD[m],
                                     "required": need, "usd_legacy": 0.0,
                                     "usd_fixed": 0.0})
        d["usd_legacy"] += a_usd
        d["usd_fixed"] += b_usd
    for d in by_market.values():
        d["usd_saved"] = round(d["usd_legacy"] - d["usd_fixed"])
        d["usd_legacy"] = round(d["usd_legacy"])
        d["usd_fixed"] = round(d["usd_fixed"])

    share = (usd_legacy - usd_fixed) / usd_legacy if usd_legacy > 0 else 0.0

    # HOW MUCH OF THIS A BUFFER ABSORBS. This is the most useful thing the gate
    # produces and it is the reason gates 2 and 4 must never be added together.
    z_sens = []
    for z, label in Z_LEVELS[:3]:
        ua = ub = 0.0
        for b in base:
            m = b["market"]
            la, _, _, _ = F.simulate(b["true_demand"], b["forecast"], b["sigma"],
                                     F.LEAD[m], z=z, horizon=legacy_horizon)
            lb, _, _, _ = F.simulate(b["true_demand"], b["forecast"], b["sigma"],
                                     F.LEAD[m], z=z, horizon=F.LEAD[m] + F.REVIEW)
            ua += la * price[b["sku"]]
            ub += lb * price[b["sku"]]
        z_sens.append({"z": z, "target": label, "usd_legacy": round(ua),
                       "usd_fixed": round(ub),
                       "share_pct": round(0.0 if ua <= 0 else (ua - ub) / ua * 100, 2)})

    sev = "HIGH" if worst > 0 and share >= 0.10 else (
        "MEDIUM" if worst > 0 else "CLEAR")

    covered = [m for m, g in gaps.items() if g["short"] == 0]
    exposed = [m for m, g in gaps.items() if g["short"] > 0]

    return _finding(
        id="G4-HORIZON",
        title="The forecast stops before the week the order is responsible for",
        severity=sev,
        diagnosis=(
            f"Required horizon is lead time plus a {F.REVIEW}-week review. "
            f"The tool provides {legacy_horizon} weeks. "
            + (f"{', '.join(exposed)} are short by up to {worst} weeks; "
               f"{', '.join(covered)} are covered. "
               "Orders in the short markets are placed on a planner's "
               "extrapolation, not on the forecast."
               if exposed else
               "Every market is covered.")),
        evidence={
            "gaps": list(gaps.values()),
            "error_by_horizon": curve,
            "by_market": list(by_market.values()),
            "usd_lost_legacy": round(usd_legacy),
            "usd_lost_fixed": round(usd_fixed),
            "z_sensitivity": z_sens,
        },
        recoverable_share=share,
        cost_to_fix="A parameter. Changing a 4 to a 7 costs nothing.",
        validation=(
            "THE AUDIT, and it needs no data science. Ask for two numbers per "
            "market: supplier lead time and forecast horizon. If horizon is "
            "less than lead plus review, orders are being placed on an "
            "extrapolation. Five minutes with a planner. Then check the "
            "pattern: losses should CONCENTRATE in the short markets. If they "
            "are spread evenly, the horizon is not the cause and this finding "
            "should be dropped."),
        caveat=(
            "A BUFFER MASKS THIS. See z_sensitivity: measured against a bare "
            "policy the shortfall costs real money, but once safety stock is "
            "sized properly it absorbs almost all of it. So gate 2 and gate 4 "
            "OVERLAP and their shares must never be added - fixing the buffer "
            "largely fixes this too. The horizon is still worth extending, "
            "because it is free and because it stops the buffer from having to "
            "cover a self-inflicted error. "
            "The size also depends on how a planner fills the gap - flat-lining "
            "the visible average is assumed here; a good planner does better "
            "and a distracted one worse. The direction does not depend on that. "
            "Finally, extending the horizon is free in software and not free in "
            "accuracy: week 7 is genuinely harder than week 1. The "
            "error-by-horizon curve is estimated from few origins and will not "
            "be smoothly monotonic - do not read a wiggle as signal."),
    )


# ---------------------------------------------------------------- gate 5
def check_coverage(series, dropped, df):
    """
    GATE 5 - is the tool forecasting series it cannot possibly forecast?

    A seasonal model needs at least one full cycle to estimate a seasonal term.
    A series younger than that has no seasonality to learn, and any model that
    appears to forecast it is fitting noise. The honest output is a refusal
    plus an analogue-based planning number, not a confident line on a chart.
    """
    period = F.SEASONAL[3]
    short = [{"market": m, "sku": s, "weeks": n} for m, s, n in dropped]
    sev = "MEDIUM" if short else "CLEAR"

    return _finding(
        id="G5-COVERAGE",
        title="Some series are too young to be forecast at all",
        severity=sev,
        diagnosis=(
            f"{len(short)} of {len(short) + len(series)} series have fewer than "
            f"{F.MIN_HISTORY} weeks against a {period}-week seasonal period. They "
            "are excluded here rather than fitted badly. A forecast for a series "
            "with less than one seasonal cycle of history is fitted noise, and "
            "reporting it with the same confidence as a three-year series is the "
            "error."
            if short else
            "Every series has enough history for the seasonal term to be "
            "estimated."),
        evidence={"excluded": short, "fitted": len(series),
                  "min_history_weeks": F.MIN_HISTORY,
                  "seasonal_period": period},
        recoverable_share=0.0,
        cost_to_fix=("A coverage rule in the pipeline, plus an analogue method "
                     "for new products - forecast from a comparable SKU's launch "
                     "curve and say so on the label."),
        validation=(
            "Backtest the incumbent tool on new-product weeks alone and compare "
            "its error there against its error on mature series. If new-product "
            "error is not materially worse, the tool is handling them and this "
            "finding does not apply."),
        caveat=(
            "This gate recovers no money directly. It prevents a confident "
            "wrong number from entering the plan, which is a different kind of "
            "value and should not be added to the other four."),
    )


# ---------------------------------------------------------------- ranking
SEV_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "CLEAR": 3}


def rank(findings):
    """
    Severity first, then recoverable share, then cost to fix.

    Deliberately NOT ranked by money alone. The cheapest fixes here are
    parameters, and a client who sees a parameter change ranked above a data
    engineering project will do the parameter change this week.
    """
    return sorted(findings,
                  key=lambda f: (SEV_RANK.get(f["severity"], 9),
                                 -f["recoverable_share"]))
