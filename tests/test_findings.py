"""
Tests for the diagnostic report and the answer-key quarantine.

These run against the committed outputs in store/ rather than refitting. A
full run is ~90 seconds of SARIMAX and the validation suite another two
minutes; a test suite nobody waits for is a test suite nobody runs. What is
asserted here are properties of the report the pipeline produced, plus one
static audit of the source that needs no run at all.

The audit is the reason this file exists. eval.py's fourth validation test
prints every reference to true_demand in the modelling files and then reports
PASS unconditionally - `p4 = True` is a literal. It is a listing, not a check.
test_the_answer_key_never_reaches_a_model_fit does the check: it parses both
modelling files and asserts that true_demand is only ever passed to the
simulator, never to anything that fits.

Run with:  python -m pytest -q
"""

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "store"

# true_demand is the answer key. These are the only callables allowed to
# receive it: the policy simulator, which scores a decision after the fact,
# and the plumbing that carries a result out.
PERMITTED_SINKS = {"F.simulate", "simulate", "out.append", "len"}

MODELLING_FILES = ("forecast.py", "checks.py")

SEVERITIES = {"HIGH", "MEDIUM", "LOW", "CLEAR"}


@pytest.fixture(scope="session")
def findings():
    return pd.read_csv(STORE / "findings.csv")


@pytest.fixture(scope="session")
def validation():
    return pd.read_csv(STORE / "validation.csv")


@pytest.fixture(scope="session")
def report():
    with open(STORE / "findings.json") as fh:
        return json.load(fh)


def sinks_receiving(path, needle="true_demand"):
    """Every callable that is passed an expression mentioning `needle`."""
    tree = ast.parse(Path(path).read_text())
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        arguments = list(node.args) + [kw.value for kw in node.keywords]
        if any(needle in ast.unparse(a) for a in arguments):
            name = ast.unparse(node.func)
            found[name] = found.get(name, 0) + 1
    return found


# --- the quarantine ------------------------------------------------------

def test_the_answer_key_never_reaches_a_model_fit():
    """
    The real version of eval.py's test 4. Parse both modelling files and
    check where true_demand actually flows. If a future edit ever hands it
    to fit(), backtest() or forecast(), this fails and names the call.
    """
    offenders = {}
    for filename in MODELLING_FILES:
        for sink, count in sinks_receiving(ROOT / filename).items():
            if sink not in PERMITTED_SINKS:
                offenders[f"{filename}:{sink}"] = count

    assert not offenders, (
        "true_demand is passed to something other than the simulator: "
        f"{offenders}. The scorecard would be restating the answer key "
        "rather than measuring against it."
    )


def test_the_forecasting_layer_never_touches_the_answer_key():
    """forecast.py fits models. It should not see true_demand at all."""
    assert sinks_receiving(ROOT / "forecast.py") == {}


# --- findings are complete ------------------------------------------------

def test_every_finding_states_what_would_disprove_it(findings):
    """
    The claim the whole report rests on. A finding with no falsification
    route is an opinion.
    """
    assert findings.validation.astype(str).str.strip().ne("").all()
    assert (findings.validation.astype(str).str.len() > 40).all()


def test_every_finding_states_what_it_costs_to_fix(findings):
    assert findings.cost_to_fix.astype(str).str.strip().ne("").all()


def test_every_finding_carries_a_caveat(findings):
    assert findings.caveat.astype(str).str.strip().ne("").all()


def test_severities_come_from_the_known_set(findings):
    assert set(findings.severity) <= SEVERITIES


def test_findings_are_ranked_by_recoverable_value(findings):
    ordered = findings.sort_values("rank")
    shares = ordered.recoverable_share_pct.tolist()
    assert shares == sorted(shares, reverse=True)
    assert ordered["rank"].tolist() == list(range(1, len(ordered) + 1))


def test_no_single_finding_claims_more_than_everything(findings):
    assert (findings.recoverable_share_pct <= 100).all()


def test_the_shares_cannot_be_summed(findings):
    """
    They overlap, and the README says so. If they ever summed to <= 100 a
    reader could add them without noticing they had double-counted, so the
    overlap is asserted rather than left as a note.
    """
    assert findings.recoverable_share_pct.sum() > 100


# --- validation is real ---------------------------------------------------

def test_a_failed_validation_withdraws_its_finding(findings, validation):
    """
    A failed test means the finding is withdrawn, not reworded. Currently
    every test passes, so this asserts the rule rather than the outcome -
    and it starts biting the day one fails.
    """
    failed = validation[validation.verdict == "FAIL"].test.tolist()
    if not failed:
        pytest.skip("no validation test is currently failing")
    reported = " ".join(findings.gate.astype(str) + " " + findings.title.astype(str)).lower()
    for name in failed:
        assert name.replace("_", " ") not in reported


def test_each_verdict_follows_from_its_own_numbers(validation):
    """
    The failure mode this catches is a hardcoded verdict - a test that
    reports PASS without comparing anything, which is what eval.py's fourth
    test does today. For the three measured tests, PASS must mean the value
    actually cleared the threshold.

    Note a threshold of 0 is legitimate: masking_vs_imputation compares a
    difference in bias points, where zero is the meaningful dividing line
    and a negative value would rightly fail.
    """
    falsifiable = validation[validation.test != "answer_key_quarantine"]
    assert len(falsifiable) >= 3

    for row in falsifiable.itertuples():
        expected = "PASS" if row.value > row.threshold else "FAIL"
        assert row.verdict == expected, (
            f"{row.test} reports {row.verdict} with value {row.value} "
            f"against threshold {row.threshold}"
        )


def test_every_validation_test_records_a_verdict(validation):
    assert set(validation.verdict) <= {"PASS", "FAIL"}


# --- scope is stated honestly --------------------------------------------

def test_excluded_series_are_reported_not_silently_dropped(report):
    excluded = pd.read_csv(STORE / "excluded_series.csv")
    assert len(excluded) == report["series_excluded"]
    assert report["series_excluded"] > 0, \
        "the coverage gate found nothing to exclude - it is not being exercised"


def test_the_report_states_the_model_it_used(report):
    assert report["model"].startswith("SARIMAX")
    assert report["holdout_weeks"] > 0
    assert report["series_fitted"] > 0
