# -*- coding: utf-8 -*-
"""
app.py - the review interface.

    python -m streamlit run app.py

WHAT THIS IS
A read-only view over what the pipeline already wrote. It does not fit, score,
simulate or recompute anything. Every number on screen came out of main.py and
was flattened by store.py, which means the figure a client sees is the figure
that was measured - they cannot drift apart.

That is a deliberate constraint, not a limitation. A dashboard that refits when
someone moves a filter is a dashboard that stalls in front of a client and, far
worse, quietly displays a number nobody validated.

WHAT IS ON EACH TAB
    Findings     the report - ranked gates, diagnosis, cost, caveat
    Evidence     the charts behind each finding
    Validation   the four tests, including the ones that could have failed
    Method       what was fitted, what was quarantined, what is synthetic

Note the escaping of the dollar sign throughout. Streamlit renders markdown,
and a bare $ starts a LaTeX block - two of them in one string silently turn
the text between into mathematics.
"""

import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Demand Forecast Health Check",
                   page_icon="📉", layout="wide")

STORE = "store"


# ------------------------------------------------------------------ loading
@st.cache_data
def load():
    def csv(name):
        p = f"{STORE}/{name}"
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    card = csv("scorecard.csv")
    return {
        "card": card.iloc[0] if len(card) else None,
        "findings": csv("findings.csv"),
        "sweep": csv("metric_sweep.csv"),
        "service": csv("service_curve.csv"),
        "hcurve": csv("horizon_curve.csv"),
        "hmarket": csv("horizon_markets.csv"),
        "hzsens": csv("horizon_zsens.csv"),
        "excluded": csv("excluded_series.csv"),
        "validation": csv("validation.csv"),
    }


D = load()
if D["card"] is None or D["findings"].empty:
    st.error(
        "No results found. Run the pipeline first:\n\n"
        "```\npython generate_data.py\npython main.py\n"
        "python store.py\npython eval.py\n```")
    st.stop()

card = D["card"]
FIND = D["findings"]

SEV_COLOUR = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡", "CLEAR": "🟢"}


# ------------------------------------------------------------------ header
st.title("Demand Forecast Health Check")
st.caption(
    f"{card.gates_run} diagnostic gates run against "
    f"{card.series_fitted} series · {card.weeks} weeks · "
    f"{card.markets} markets · {card.skus} SKUs · "
    f"holdout {card.holdout_weeks} weeks"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Gates fired", f"{card.gates_high} HIGH",
          f"{card.gates_clear} clear of {card.gates_run}",
          delta_color="off")
k2.metric("Largest single fix", f"{card.largest_share_pct:.0f}%",
          str(card.largest_gate), delta_color="off")
k3.metric("Forecast bias today", f"{card.bias_naive_pct:.1f}%",
          f"{card.bias_repaired_pct:.1f}% after repair", delta_color="off")
k4.metric("Weeks stocked out", f"{card.censoring_rate_pct:.1f}%",
          "of live weeks in history", delta_color="off")

st.info(
    "**The shares below do not add up, and must not be summed.** Censored data "
    "and the accuracy metric both push the forecast down, so fixing either "
    "helps and fixing both is not the sum of the two. A properly sized buffer "
    "absorbs most of the horizon shortfall, so gate 2 largely contains gate 4. "
    "The defensible headline is the largest single share, with a note that the "
    "others compound it.",
    icon="⚠️")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.subheader("Scope")
    sev = st.multiselect("Severity", ["HIGH", "MEDIUM", "LOW", "CLEAR"],
                         default=["HIGH", "MEDIUM", "LOW", "CLEAR"])
    st.divider()

    st.subheader("What was fitted")
    st.write(f"**Model** `{card.model}`")
    st.write(f"**Series fitted** {card.series_fitted}")
    st.write(f"**Series excluded** {card.series_excluded}")
    st.caption(
        "Excluded means too little history for the seasonal term to be "
        "estimated at all. They are refused, not fitted badly - a forecast "
        "built on less than one seasonal cycle is fitted noise.")

    st.divider()
    st.subheader("Validation")
    if not D["validation"].empty:
        V = D["validation"]
        for _, r in V.iterrows():
            mark = "✅" if r.verdict == "PASS" else "❌"
            st.write(f"{mark} {r.test.replace('_', ' ')}")
        st.caption(
            "Three of these can fail. If one does, the matching finding is "
            "withdrawn from the report rather than reworded.")
    else:
        st.caption("eval.py has not been run.")

    st.divider()
    st.caption(f"Generated {card.generated_utc} · "
               f"pipeline runtime {card.runtime_seconds:.0f}s")

view = FIND[FIND.severity.isin(sev)] if sev else FIND


# ------------------------------------------------------------------ tabs
t_find, t_evid, t_val, t_meth = st.tabs(
    ["Findings", "Evidence", "Validation", "Method"])


# ---------------------------------------------------------------- findings
with t_find:
    if view.empty:
        st.warning("No gates match that severity filter.")
    else:
        st.dataframe(
            view[["rank", "gate", "severity", "recoverable_share_pct",
                  "title", "cost_to_fix"]].rename(columns={
                      "recoverable_share_pct": "recovers %",
                      "cost_to_fix": "cost to fix"}),
            hide_index=True, width="stretch")

        st.divider()
        for _, r in view.iterrows():
            with st.expander(
                    f"{SEV_COLOUR.get(r.severity,'')}  {r.gate}  ·  "
                    f"{r.title}  ·  recovers {r.recoverable_share_pct:.1f}%",
                    expanded=(r["rank"] == 1)):
                st.markdown(f"**Diagnosis**  \n{r.diagnosis}")
                c1, c2 = st.columns(2)
                c1.metric("Recoverable share",
                          f"{r.recoverable_share_pct:.1f}%",
                          "of stockout loss on the holdout", delta_color="off")
                c2.markdown(f"**Cost to fix**  \n{r.cost_to_fix}")
                st.markdown(f"**How to prove this wrong**  \n{r.validation}")
                st.markdown(f"**What this does not establish**  \n{r.caveat}")


# ---------------------------------------------------------------- evidence
with t_evid:
    st.subheader("G1 · the metric asks for a lower forecast")
    if not D["sweep"].empty:
        S = D["sweep"]
        st.caption(
            "One forecast, multiplied by k. Nothing changes but the ruler. "
            "If MAPE were symmetric its minimum would sit at k = 1.00.")
        st.line_chart(S.set_index("k")[["mape", "wmape"]], height=260)
        c1, c2 = st.columns(2)
        c1.metric("MAPE picks", f"k = {card.k_mape:.2f}")
        c2.metric("WMAPE picks", f"k = {card.k_wmape:.2f}")
        st.caption(
            "MAPE divides by the actual, so under-forecasting to zero scores "
            "at worst 100% while over-forecasting by 5x scores 400%. The "
            "punishment for being low has a floor and the punishment for "
            "being high does not.")
        with st.expander("the full sweep"):
            st.dataframe(S, hide_index=True, width="stretch")

    st.divider()
    st.subheader("G2 · what a service level buys")
    if not D["service"].empty:
        SV = D["service"]
        st.caption(
            "Same forecast throughout. What changes is how much of its error "
            "the ordering policy is willing to absorb.")
        st.line_chart(SV.set_index("z")[["fill_rate_pct", "weeks_ok_pct"]],
                      height=260)
        st.dataframe(SV, hide_index=True, width="stretch")
        st.warning(
            "This gate's share is an **upper bound**. The simulation has fixed "
            "lead times, no supplier shortfalls, no minimum order quantities "
            "and no shelf constraint. Every one of those pushes the real "
            "recovery lower. Quote the mechanism, treat the level as a ceiling.",
            icon="⚠️")

    st.divider()
    st.subheader("G4 · the forecast stops short of the decision")
    if not D["hmarket"].empty:
        HM = D["hmarket"]
        st.caption(
            "An order placed today lands in `lead` weeks and covers demand "
            "until the next one lands. Required horizon is lead + review. "
            "Every dollar of loss should sit where that gap is - if it were "
            "spread evenly across markets, this diagnosis would be wrong.")
        st.dataframe(HM, hide_index=True, width="stretch")
        st.bar_chart(HM.set_index("market")["usd_saved"], height=240)

    if not D["hzsens"].empty:
        st.markdown("**How much of it a buffer absorbs**")
        st.dataframe(D["hzsens"], hide_index=True, width="stretch")
        st.caption(
            "The share falls to zero once z rises above zero. Gates 2 and 4 "
            "OVERLAP - fixing the buffer largely fixes this too, which is "
            "precisely why the shares cannot be added.")

    if not D["hcurve"].empty:
        st.markdown("**Forecast error at each horizon, measured separately**")
        st.line_chart(D["hcurve"].set_index("horizon")["wmape_pct"], height=240)
        st.caption(
            "A team reporting one-step accuracy is reporting accuracy at a "
            "horizon that drives no decision. This curve is estimated from few "
            "origins and will not be smoothly monotonic - do not read a wiggle "
            "as signal.")

    st.divider()
    st.subheader("G5 · series too young to forecast")
    if not D["excluded"].empty:
        st.dataframe(D["excluded"], hide_index=True, width="stretch")
        st.caption(
            "Fewer weeks of history than one seasonal cycle. Refused rather "
            "than fitted. This recovers no money - it prevents a confident "
            "wrong number entering the plan, which is a different kind of "
            "value and should not be added to the others.")
    else:
        st.success("Every series has enough history to fit.")


# -------------------------------------------------------------- validation
with t_val:
    st.subheader("Four tests, three of which can fail")
    if D["validation"].empty:
        st.warning("Run `python eval.py` to populate this tab.")
    else:
        V = D["validation"]
        st.dataframe(V, hide_index=True, width="stretch")
        n_fail = int((V.verdict == "FAIL").sum())
        if n_fail:
            st.error(
                f"{n_fail} test(s) failed. A failed test means the matching "
                "finding should be **withdrawn** from the report, not "
                "reworded.")
        else:
            st.success("All tests passed.")

        st.markdown("""
**The placebo** — take weeks that did *not* stock out, censor them artificially,
refit, and repair. You know the truth for those weeks because you withheld it
yourself, so this runs on a real client's data with no answer key. The measure
is a *residual*, not a movement: both the damaged and undamaged series are
repaired and compared, so pre-existing censoring cancels out. An earlier version
measured the movement instead and reported "286% recovered", which is not a
number that can exist.

**The intuitive fix** — everyone proposes removing the stockout weeks. There are
two ways to do it and they are not the same. Dropping the *rows* re-indexes
everything after the gap, so week t−52 is no longer a year earlier; the bias can
improve while the seasonal structure is being destroyed. Masking the *values*
keeps the alignment, and on that honest comparison the intuitive fix makes the
bias worse.

**The market pattern** — the horizon diagnosis predicts loss concentrates where
lead time exceeds the forecast horizon. If it were spread evenly across markets,
that finding would be wrong and would be dropped. This test could have failed.

**The answer key** — a source audit confirming that `true_demand` never reaches
a model fit. It enters only through the evaluation harness, after the fact.
""")


# ------------------------------------------------------------------ method
with t_meth:
    st.subheader("What was actually done")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", str(card.model))
    c2.metric("Holdout", f"{card.holdout_weeks} weeks")
    c3.metric("Pipeline runtime", f"{card.runtime_seconds:.0f}s")

    st.markdown("""
**The model.** SARIMAX on log(shipments), with the event calendar as exogenous
regressors. The log is there because variance rises with the level, which makes
the seasonal swing multiplicative; the log makes it additive. One seasonal
difference removes the annual cycle. No seasonal AR or MA term survived — three
years is very little to estimate them from, and saying so is more honest than
fitting them anyway.

**Why exogenous regressors and not a bigger seasonal term.** Ramadan moves about
eleven days earlier each Gregorian year, so it is *not* a 52-week seasonal
effect and a seasonal term cannot represent it. It has to arrive as a regressor.
That is the entire SARIMA → SARIMAX step and it is the one modelling choice here
that is not cosmetic.

**What this check does not recommend.** It does not recommend a model. Nothing
in the report says "use a neural network". Every finding is a fault in the
process *around* the forecast — the metric, the buffer, the training data, the
horizon, the coverage rule — because that is where the money nearly always is,
and because those fixes cost days rather than quarters.

**The answer key.** `true_demand` exists in this dataset so the evaluation can
show what a censored history hides. A real engagement never has it, which is
exactly why every validation test above is built to run without it. Nothing that
fits a model is permitted to read it, and `eval.py` audits the source to confirm.

**The data.** Synthetic, generated by `generate_data.py` from a fixed seed, so
every figure reproduces exactly. The faults are ones I constructed deliberately:
censored demand, a moving religious holiday, a promotional calendar, a new
product with no history, and a lead time longer than the forecast horizon. On
real data the *levels* will differ. What transfers is the method — which gate
fires, what evidence it produces, and how to prove it wrong.
""")

    st.divider()
    st.caption(
        "Pipeline: generate_data.py → main.py → store.py → eval.py → app.py. "
        "The app reads only what those wrote; it computes nothing at display "
        "time, so the number on screen is the number that was measured.")
