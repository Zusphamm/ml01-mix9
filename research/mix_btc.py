"""Low-correlation mix of the 39 SINGLES_OS10 alphas + a small BTC filter, as a live-tradable top-K book
(user request 2026-10-19).

  python mix_btc.py prereg | is | os

Selection (IS 2021-01 .. 2023-12 daily net of each alpha, out/singles_report_daily.parquet; nothing else):
  CLUST   average-linkage clusters on 1 - corr, cut at 0.5 (alphas with corr >= 0.5 merge); the best IS Sharpe per cluster
  GREEDY  walk down the IS Sharpe ranking, keep an alpha if its |corr| with every kept one is <= 0.4, at most 8
Book: composite z = mean of member z (live_options._z: sign x winsor(zscore, 3) on the PIT top-200 basket, NaN skipped),
K highest +0.5/K, K lowest -0.5/K, K in {15, 30}; live_options.pnl (tiered cost, real funding), reported at 1.27x gross.
BTC filter (known at the close of t, applied to the weights of t):
  none
  trend   BTC close < its 50 d average -> book x 0.5
  hedge   subtract the book's BTC beta (sum of w_i x 60 d daily beta_i to BTC) with a BTCUSDT leg
Trials 2 x 2 x 3 = 12; primary CLUST_K15_trend. Pass (IS): Sharpe >= LIVE (2.11) and max DD better than LIVE (-28.8%);
the filter earns its place (primary Sharpe >= CLUST_K15_none); DSR(N 20 = 12 + 8 earlier live-book variants) >= 0.95.
OS 2024-01 .. 2026-06 once only on a pass, also WFE >= 0.6. The 39 were picked with OS >= 1.0, so OS is not a clean
holdout for any mix of them; the clean test is forward paper. BLACKBOX not read.
"""
import datetime as dt, itertools, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import average, fcluster
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
import trial_screen as S, trial_screen_os as SO, run_grammar_daemon as M, live_options as LO   # noqa: E402
from newperp_fade import dsr                                                                  # noqa: E402

OUT = S.OUT / "mix_btc"
PRE = OUT / "prereg.json"
LIVE, LEV, N_FAMILY = "cond|illiq60|dist_high60|low|s0", 2000 / 1573.9, 20
TRIALS = {f"{s}_K{k}_{f}": (s, k, f) for s, k, f in itertools.product(("CLUST", "GREEDY"), (15, 30), ("none", "trend", "hedge"))}
PRIMARY = "CLUST_K15_trend"


def is_daily():
    cfg = json.loads(LO.CFG.read_text())["alphas"]
    R = pd.read_parquet(S.OUT / "singles_report_daily.parquet")[[a["key"] for a in cfg]]
    R = R[(R.index >= S.IS0) & (R.index < S.OS0)]
    assert R.index.max() < S.OS0
    return R, {a["key"]: int(a["sign"]) for a in cfg}


def select(R):
    sr = R.mean() / R.std() * math.sqrt(365)
    C = R.corr()
    Z = average(squareform((1 - C).clip(lower=0).values, checks=False))
    lab = pd.Series(fcluster(Z, t=0.5, criterion="distance"), index=R.columns)
    clust = sorted(sr.groupby(lab).idxmax().tolist())
    greedy = []
    for k in sr.sort_values(ascending=False).index:
        if all(abs(C.at[k, g]) <= 0.4 for g in greedy):
            greedy.append(k)
        if len(greedy) == 8:
            break
    return dict(CLUST=clust, GREEDY=greedy), sr, C, lab


def load(until):
    SO.load_full(until=until)
    P, close = S.G["P"], S.G["P"]["close"]
    for f in M.EXTRA:
        q = M.PIT1D / f"{f}.parquet"
        if q.exists():
            P[f] = pd.read_parquet(q).astype(float).reindex_like(close)
    S.G["env"] = M.build_env(P)
    return P


def books(P, sets, sign):
    close = P["close"]
    Z = {k: LO._z((k, sign[k]))[1] for k in sorted({k for v in sets.values() for k in v} | {LIVE})}
    comp = {s: sum(Z[k].fillna(0) for k in ks) / sum(Z[k].notna().astype(int) for k in ks).replace(0, np.nan) for s, ks in sets.items()}
    btc = close["BTCUSDT"]
    trend = (btc < btc.rolling(50, min_periods=40).mean()).reindex(close.index).fillna(False)
    ret = close.pct_change(fill_method=None)
    beta = ret.rolling(60, min_periods=40).cov(ret["BTCUSDT"]).div(ret["BTCUSDT"].rolling(60, min_periods=40).var(), axis=0).clip(-1, 4)
    out = {"LIVE": LO.topk(Z[LIVE], 15)}
    for name, (s, k, f) in TRIALS.items():
        W = LO.topk(comp[s], k)
        if f == "trend":
            W = W.mul(np.where(trend, 0.5, 1.0), axis=0)
        elif f == "hedge":
            b = (W * beta.reindex_like(W)).sum(axis=1)
            W = W.copy()
            W["BTCUSDT"] = W["BTCUSDT"] - b
        out[name] = W
    return out


def smb(P):
    fwd = P["close"].pct_change(fill_method=None).shift(-1)
    rk = P["quote_volume"].rolling(30, min_periods=10).mean().rank(axis=1, ascending=False)
    return fwd.where((rk > 100) & (rk <= 200)).mean(1) - fwd.where(rk <= 20).mean(1)


def stats(r, f, live):
    eq, s30 = (1 + r).cumprod(), r.rolling(30).sum()
    return dict(sharpe=round(float(r.mean() / r.std() * math.sqrt(365)), 2), ret_yr=round(float(r.mean() * 365 * 100), 1),
                maxdd=round(float((eq / eq.cummax() - 1).min() * 100), 1), p30neg=round(float((s30 < 0).mean()), 2),
                worst30=round(float(s30.min() * 100), 1), corr_smb=round(float(r.corr(f.reindex(r.index))), 2),
                corr_live=round(float(r.corr(live)), 2),
                **{f"y{y}": round(float(r[r.index.year == y].mean() / r[r.index.year == y].std() * math.sqrt(365)), 2) for y in sorted(set(r.index.year))})


def run(seg):
    pre = json.loads(PRE.read_text())
    if seg == "os":
        assert not (OUT / "os.json").exists(), "OS already run once"
        assert json.loads((OUT / "is.json").read_text())["pass"], "IS did not pass: no OS"
        with open(S.OUT / "os_access.log", "a") as fh:
            fh.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} mix_btc.py OS run (2024-01..2026-06, once) of 12 mixes; "
                     f"components already OS-selected; requested by user\n")
    sets, sign = pre["sets"], json.loads(json.dumps(is_daily()[1]))
    lo, hi = (S.IS0, S.OS0) if seg == "is" else (S.OS0, S.BLACKBOX0)
    P = load(S.OS0 if seg == "is" else S.BLACKBOX0)
    if seg == "is":
        assert P["close"].index.max() < S.OS0
    f, B = smb(P), None
    B = books(P, sets, sign)
    D = {n: (LO.pnl(W).iloc[:-1].net * LEV) for n, W in B.items()}
    D = {n: r[(r.index >= lo) & (r.index < hi)] for n, r in D.items()}
    res = {n: stats(r, f, D["LIVE"]) for n, r in D.items()}
    for n, v in res.items():
        print(f"{n:18s}", json.dumps(v), flush=True)
    M_ = pd.DataFrame({n: D[n] for n in TRIALS})
    srs = [M_[c].mean() / M_[c].std() for c in M_]
    p, live = res[PRIMARY], res["LIVE"]
    out = dict(seg=seg, sets=sets, results=res, primary=PRIMARY, dsr=dsr(M_[PRIMARY], srs, N_FAMILY))
    if seg == "is":
        out["checks"] = dict(sharpe_ge_live=p["sharpe"] >= live["sharpe"], maxdd_better=p["maxdd"] > live["maxdd"],
                             filter_helps=p["sharpe"] >= res["CLUST_K15_none"]["sharpe"], dsr_95=out["dsr"] >= 0.95)
    else:
        isr = json.loads((OUT / "is.json").read_text())["results"][PRIMARY]["sharpe"]
        out["wfe"] = round(p["sharpe"] / isr, 2)
        out["checks"] = dict(wfe_06=out["wfe"] >= 0.6, sharpe_ge_live=p["sharpe"] >= live["sharpe"], maxdd_better=p["maxdd"] > live["maxdd"])
    out["pass"] = all(out["checks"].values())
    (OUT / f"{seg}.json").write_text(json.dumps(out, indent=1, default=float))
    pd.DataFrame(D).to_parquet(OUT / f"{seg}_daily.parquet")
    print(json.dumps({k: out[k] for k in ("dsr", "checks", "pass")}, default=str))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "prereg":
        assert not PRE.exists(), "prereg already frozen"
        R, sign = is_daily()
        sets, sr, C, lab = select(R)
        info = {s: [dict(key=k, is_sharpe=round(float(sr[k]), 2), cluster=int(lab[k])) for k in ks] for s, ks in sets.items()}
        mc = {s: round(float(C.loc[ks, ks].where(~np.eye(len(ks), dtype=bool)).abs().max().max()), 2) for s, ks in sets.items()}
        PRE.write_text(json.dumps(dict(frozen_at=dt.datetime.now(dt.timezone.utc).isoformat(), sets=sets, members=info,
                                       max_pair_corr=mc, n_clusters=int(lab.nunique()), trials=TRIALS, primary=PRIMARY,
                                       n_family=N_FAMILY, spec="mix_btc.py docstring"), indent=1, default=str))
        print(json.dumps(dict(sets=info, max_pair_corr=mc, n_clusters=int(lab.nunique())), indent=1))
    else:
        run(sys.argv[1])
