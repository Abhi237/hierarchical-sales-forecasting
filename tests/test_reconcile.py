from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hier_forecast import reconcile as R
from hier_forecast.data import load_panel, summing_matrix


@pytest.fixture(scope="module")
def S(con):
    panel = load_panel(con, "hier")
    S, _ = summing_matrix(con, panel.node_ids)
    return S, panel


def test_S_shape_and_structure(S):
    S, panel = S
    assert S.shape == (48, 12)
    assert set(np.unique(S)) == {0.0, 1.0}
    assert S[0].sum() == 12                      # total row
    assert R.bottom_index(S).shape == (12,)


def test_actuals_are_coherent_under_S(S):
    S, panel = S
    b = panel.Y[:, R.bottom_index(S)]
    assert np.allclose(b @ S.T, panel.Y)


@pytest.mark.parametrize("method", ["bu", "ols", "wls_struct"])
def test_G_is_a_left_inverse_of_S(S, method):
    S, _ = S
    G = R.g_matrix(S, method)
    assert np.allclose(G @ S, np.eye(S.shape[1]))


def test_mint_and_wls_var_G_left_inverse(S):
    S, panel = S
    rng = np.random.default_rng(0)
    E = rng.normal(size=(40, S.shape[0]))
    W = R.shrink_cov(E)
    for m in ("wls_var", "mint"):
        assert np.allclose(R.g_matrix(S, m, W) @ S, np.eye(S.shape[1]))


@pytest.mark.parametrize("method", R.METHODS)
def test_reconciled_forecasts_are_coherent(S, method):
    S, panel = S
    rng = np.random.default_rng(1)
    F = panel.Y[-4:].T * (1 + 0.1 * rng.standard_normal((S.shape[0], 4)))
    W = R.shrink_cov(rng.normal(size=(30, S.shape[0])) * panel.Y.mean(0))
    out = R.reconcile_matrix(F, S, method, W)
    b = out[R.bottom_index(S)]
    assert np.allclose(S @ b, out)


@pytest.mark.parametrize("method", R.METHODS)
def test_coherent_input_is_unchanged(S, method):
    S, panel = S
    F = panel.Y[-4:].T
    rng = np.random.default_rng(2)
    W = R.shrink_cov(rng.normal(size=(30, S.shape[0])))
    assert np.allclose(R.reconcile_matrix(F, S, method, W), F)


def test_bottom_up_uses_bottom_forecasts_only(S):
    S, panel = S
    F = panel.Y[-2:].T.copy()
    F[0] += 1e6                                     # corrupt the total forecast
    out = R.reconcile_matrix(F, S, "bu")
    bi = R.bottom_index(S)
    assert np.allclose(out[bi], F[bi])
    assert not np.isclose(out[0, 0], F[0, 0])


def test_ols_is_idempotent_projection(S):
    S, panel = S
    rng = np.random.default_rng(3)
    F = rng.normal(size=(S.shape[0], 3)) + 100
    once = R.reconcile_matrix(F, S, "ols")
    twice = R.reconcile_matrix(once, S, "ols")
    assert np.allclose(once, twice)


def test_shrink_cov_is_symmetric_positive_definite():
    rng = np.random.default_rng(4)
    E = rng.normal(size=(20, 30)) @ rng.normal(size=(30, 30))
    W = R.shrink_cov(E)
    assert np.allclose(W, W.T)
    assert np.linalg.eigvalsh(W).min() > 0
    assert np.allclose(np.diag(W), np.diag(E.T @ E / 20))


def test_coherence_gap_zero_for_coherent_and_positive_otherwise(S):
    S, panel = S
    b = panel.Y[-1][R.bottom_index(S)]
    rows = lambda v: pd.DataFrame({"model": "m", "origin": 1, "h": 1, "node_idx": np.arange(len(v)), "yhat": v})  # noqa: E731
    g0 = R.coherence_gap(rows(S @ b), S, panel.level_ids)
    assert g0["mean_abs_rel_gap"].max() < 1e-12
    v = S @ b
    v[0] *= 1.1
    g1 = R.coherence_gap(rows(v), S, panel.level_ids)
    assert g1[g1["level"] == 1]["mean_abs_rel_gap"].iloc[0] > 0.05


def test_python_reconciled_models_exist_and_are_coherent(built, con):
    from hier_forecast.config import rpath
    panel = load_panel(con, "hier")
    S, _ = summing_matrix(con, panel.node_ids)
    f = pd.read_parquet(rpath(built, built["data"]["processed_dir"], "forecasts_hier_all.parquet"))
    sub = f[f["model"] == "py_ets_mint"]
    assert len(sub) > 0
    one = sub[(sub["origin"] == sub["origin"].max()) & (sub["h"] == 1)].sort_values("node_idx")
    v = one["yhat"].to_numpy()
    assert np.allclose(S @ v[R.bottom_index(S)], v)


def test_reconciliation_only_in_validation_and_test(built, con):
    from hier_forecast.config import origins, rpath
    panel = load_panel(con, "hier")
    o = origins(built, panel.T)
    f = pd.read_parquet(rpath(built, built["data"]["processed_dir"], "forecasts_hier_all.parquet"))
    py = f[f["model"].str.startswith("py_")]
    assert set(py["origin"]) <= set(o["validation"] + o["test"])
