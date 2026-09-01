# Demand Forecast Health Check Agent

An agent that reads a demand history and tells you **why** the forecast keeps
stocking out — not by building a bigger model, but by testing for the small
number of well-understood faults that actually cause it.

A forecast that misses is almost never fixed by a better algorithm. It is fixed
by finding which of these is present: the accuracy metric is rewarding
under-forecasting, orders are sized with no buffer, the training data records
shipments instead of demand, the forecast stops before the week the order is
responsible for, or the tool is confidently forecasting series too young to
forecast at all.

Five gates run against the history and return a ranked report — each finding
with its evidence, what fraction of the loss it recovers, what it costs to fix,
**and a test the client can run that would prove it wrong.**

Built on a synthetic 3-year weekly history: 156 weeks, 3 markets, 4 SKUs. No
client data is used anywhere in this repository.

---

## What it finds on the demo dataset

| Rank | Gate | Severity | Recovers | Cost to fix |
|---|---|---|---|---|
| 1 | Orders sized on the point forecast, no buffer | HIGH | 90.3% | A parameter in the replenishment rule |
| 2 | The accuracy metric rewards under-forecasting | HIGH | 55.7% | One line in the scoring script |
| 3 | Model trained on shipments, not demand | HIGH | 17.5% | A stockout flag, then an EM loop |
| 4 | Forecast stops short of the decision week | MEDIUM | 3.5% | A parameter. Changing a 4 to a 7 |
| 5 | Series too young to forecast at all | MEDIUM | — | A coverage rule in the pipeline |

**The top two fixes cost nothing.** No new model, no new data, no new system.
That ranking is the product.

**These shares overlap and must not be summed.** Censored data biases the
forecast down and the metric then rewards pushing it down further. A properly
sized buffer absorbs almost all of the horizon shortfall, so gate 2 largely
contains gate 4. The defensible headline is the largest single share with a
note that the others compound it — anyone adding them to reach a bigger number
is overselling, and the client's finance team will find it.

---

## The part that matters: it can fail

Four validation tests, three of which can return FAIL. A failed test means the
matching finding is **withdrawn** from the report, not reworded.

| Test | Result | What it establishes |
|---|---|---|
| Placebo | 63.7% | Censoring artificially induced on clean weeks; the repair removed two thirds of it |
| Intuitive fix | +1.3 pts | Masking stockout weeks makes bias *worse*; imputation makes it better |
| Market pattern | 97.7% | Loss concentrates in the markets short of horizon, not spread evenly |
| Answer key | PASS | Source audit: `true_demand` never reaches a model fit |

**The placebo needs no answer key**, which is why it runs on a real client's
data. Take weeks that did *not* stock out, censor them artificially, refit,
repair, and measure what is left. You know the truth for those weeks because
you withheld it yourself.

**The market pattern test could have failed and didn't.** The horizon diagnosis
predicts loss concentrates where lead time exceeds forecast horizon. It does —
97.7% against 33.3% for an even spread. Indonesia carries the whole loss;
Singapore, whose lead time already fits inside the horizon, carries zero. If a
bigger model were the answer, all three markets would be losing money together.

---

## Two mistakes this build caught in itself

Both are documented in the code rather than quietly fixed, because they are the
kind of error that flatters a result.

**The placebo first reported "286% recovered"** — an impossible number. These
series already contain real censoring, so the repair was fixing both the
induced damage and the pre-existing damage, and all of it was being credited to
the placebo. The fix is to repair both the damaged and undamaged versions and
compare the *residual*, so pre-existing censoring cancels. It now reports 64%.

**Dropping stockout rows looks like the best fix on the table.** It isn't.
Removing rows from a series with a 52-week seasonal term re-indexes everything
after the gap — week t−52 is no longer a year earlier. The bias improves
because the seasonal structure has been destroyed. Masking the *values* keeps
the alignment, and on that honest comparison the intuitive fix makes things
worse.

---

## The model

    SARIMAX(1,0,1)(0,1,0,52) on log(shipments), event calendar as exog

- **log** — variance rises with the level, so the log makes the seasonal swing
  additive rather than multiplicative
- **(1,0,1)** — ACF and PACF both tail off rather than cutting off, the
  signature of a mixed ARMA term
- **(0,1,0,52)** — one seasonal difference. No seasonal AR or MA term survived;
  three years is very little to estimate them from, and saying so is more
  honest than fitting them anyway
- **exog** — Ramadan moves ~11 days earlier each Gregorian year, so it is *not*
  a 52-week seasonal effect and a seasonal term cannot represent it. It has to
  arrive as a regressor. That is the entire SARIMA → SARIMAX step and the one
  modelling choice here that is not cosmetic

---

## Run it

    pip install -r requirements.txt
    python -m streamlit run app.py

`data/` and `store/` are committed, so the app runs straight from a clone. It
computes nothing at display time — it reads the CSVs the pipeline already
wrote, which means the number on screen is the number that was measured.

## Rebuild from scratch

    python generate_data.py     # synthetic history, fixed seed
    python main.py              # fit, run five gates, rank      ~90s
    python store.py             # flatten into the app's tables
    python eval.py              # the four validation tests      ~2min

Only `eval.py` may read `true_demand`. Nothing that fits a model can see it,
and `eval.py` audits the source of `forecast.py` and `checks.py` to confirm.

---

## Layout

    forecast.py       shared modelling layer - fit, backtest, EM, simulate
    checks.py         the five diagnostic gates. This is the agent
    main.py           run the gates, rank, write findings.json
    store.py          flatten into the tables the app reads
    eval.py           the falsifiability layer
    app.py            Streamlit review interface
    generate_data.py  the synthetic history and the faults built into it
    data/             demand, events, promos, SKU master
    store/            findings and evidence, committed so the app runs on clone

`forecast.py` and `checks.py` are **modules** — they are imported, never run
directly. The other four are scripts.

---

## Honest notes

The dataset is synthetic, generated from a fixed seed, so every figure above
reproduces exactly. The faults were constructed deliberately: censored demand,
a moving religious holiday, a promotional calendar, a new product with no
history, and a lead time longer than the forecast horizon.

On real data the **levels** will differ, and the safety-stock share in
particular is an **upper bound** — the simulation has fixed lead times, no
supplier shortfalls, no minimum order quantities and no shelf constraint, and
every one of those pushes real recovery lower.

What transfers is the method: which gate fires, what evidence it produces, and
how the client can prove it wrong.

---

© 2026 Debadrita Choudhury. Published for evaluation — see [LICENSE](LICENSE).
Not licensed for reuse.
