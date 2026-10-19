"""Confirmatory OS of one low-correlation mix (user choice 2026-10-19, after mix_btc.py IS).

  python mix_os.py prereg | os

Trial: CLUST_K15_none from mix_btc.py (9 alphas, best IS Sharpe per correlation cluster; composite z; top-15/15; no BTC
filter). Chosen after seeing the 12 IS rows (IS Sharpe 3.26 vs LIVE 2.12; the BTC filters did not help), so this is a
post-hoc pick from 12. OS 2024-01 .. 2026-06 run once, only this book and the LIVE reference are computed. The 39
components were themselves picked with OS >= 1.0: OS is not clean; forward paper is the real test. BLACKBOX not read.
Pass: WFE (OS / IS Sharpe) >= 0.6, OS Sharpe >= LIVE OS Sharpe, OS max DD better than LIVE's.
"""
import datetime as dt, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
import trial_screen as S, live_options as LO   # noqa: E402
import mix_btc as MB                           # noqa: E402

OUT = S.OUT / "mix_os"
PRE = OUT / "prereg.json"
TRIAL = "CLUST_K15_none"


def run():
    assert PRE.exists() and not (OUT / "os.json").exists(), "prereg missing or OS already run once"
    pre = json.loads(PRE.read_text())
    with open(S.OUT / "os_access.log", "a") as fh:
        fh.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} mix_os.py OS run (2024-01..2026-06, once) of {TRIAL} "
                 f"(post-hoc pick of 12 IS rows; components already OS-selected); requested by user\n")
    sign = MB.is_daily()[1]
    P = MB.load(S.BLACKBOX0)
    ks = pre["members"]
    Z = {k: LO._z((k, sign[k]))[1] for k in sorted(set(ks) | {MB.LIVE})}
    comp = sum(Z[k].fillna(0) for k in ks) / sum(Z[k].notna().astype(int) for k in ks).replace(0, np.nan)
    D = {"LIVE": LO.pnl(LO.topk(Z[MB.LIVE], 15)).iloc[:-1].net * MB.LEV, TRIAL: LO.pnl(LO.topk(comp, 15)).iloc[:-1].net * MB.LEV}
    D = {n: r[(r.index >= S.OS0) & (r.index < S.BLACKBOX0)] for n, r in D.items()}
    f = MB.smb(P)
    res = {n: MB.stats(r, f, D["LIVE"]) for n, r in D.items()}
    isr = json.loads((MB.OUT / "is.json").read_text())["results"][TRIAL]["sharpe"]
    p, live = res[TRIAL], res["LIVE"]
    out = dict(results=res, is_sharpe=isr, wfe=round(p["sharpe"] / isr, 2), members=ks)
    out["checks"] = dict(wfe_06=out["wfe"] >= 0.6, sharpe_ge_live=p["sharpe"] >= live["sharpe"], maxdd_better=p["maxdd"] > live["maxdd"])
    out["pass"] = all(out["checks"].values())
    (OUT / "os.json").write_text(json.dumps(out, indent=1, default=float))
    pd.DataFrame(D).to_parquet(OUT / "os_daily.parquet")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "prereg":
        assert not PRE.exists(), "prereg already frozen"
        members = json.loads(MB.PRE.read_text())["sets"]["CLUST"]
        PRE.write_text(json.dumps(dict(frozen_at=dt.datetime.now(dt.timezone.utc).isoformat(), trial=TRIAL, members=members,
                                       k=15, lev=MB.LEV, spec="mix_os.py docstring"), indent=1))
        print("prereg frozen", TRIAL, len(members), "members")
    else:
        assert sys.argv[1] == "os"
        run()
