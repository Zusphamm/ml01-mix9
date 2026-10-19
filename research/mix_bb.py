"""Blackbox run of the MIX9 book (user request 2026-10-19 "chạy bb đi"), once.

  python mix_bb.py

MIX9 = mix_os.py members (CLUST_K15_none), composite z, top-15/15, live engine, 1.27x gross; LIVE as reference.
BLACKBOX window 2026-07-01 .. last complete bar (unsealed with until=None, logged). The 39 components each had their
blackbox read before (os10_blackbox.py, to 2026-10-09) and LIVE's book too (blend_live.py, to 10-14); this mix had not.
Report only: BB Sharpe > 0 and max DD better than LIVE's are the checks; no parameter changes follow from it.
"""
import datetime as dt, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
import trial_screen as S, live_options as LO   # noqa: E402
import mix_btc as MB                           # noqa: E402

OUT = S.OUT / "mix_bb"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "bb.json").exists(), "blackbox already run once"
    with open(S.OUT / "os_access.log", "a") as fh:
        fh.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} mix_bb.py BLACKBOX (until=None, once) of MIX9 CLUST_K15_none "
                 f"+ LIVE reference, requested by user\n")
    ks = json.loads((S.OUT / "mix_os/prereg.json").read_text())["members"]
    sign = MB.is_daily()[1]
    P = MB.load(None)
    Z = {k: LO._z((k, sign[k]))[1] for k in sorted(set(ks) | {MB.LIVE})}
    comp = sum(Z[k].fillna(0) for k in ks) / sum(Z[k].notna().astype(int) for k in ks).replace(0, np.nan)
    D = {"LIVE": LO.pnl(LO.topk(Z[MB.LIVE], 15)).iloc[:-1].net * MB.LEV, "MIX9": LO.pnl(LO.topk(comp, 15)).iloc[:-1].net * MB.LEV}
    D = {n: r[r.index >= S.BLACKBOX0] for n, r in D.items()}
    f = MB.smb(P)
    res = {n: MB.stats(r, f, D["LIVE"]) for n, r in D.items()}
    m = pd.DataFrame(D)
    out = dict(window=[str(m.index.min().date()), str(m.index.max().date())], days=len(m), results=res,
               monthly={str(k): v for k, v in (m.groupby(m.index.to_period("M")).sum() * 100).round(2).to_dict("index").items()},
               last_days={str(k.date()): v for k, v in (m.tail(8) * 100).round(2).to_dict("index").items()})
    out["checks"] = dict(sharpe_pos=res["MIX9"]["sharpe"] > 0, maxdd_better=res["MIX9"]["maxdd"] > res["LIVE"]["maxdd"])
    (OUT / "bb.json").write_text(json.dumps(out, indent=1, default=float))
    m.to_parquet(OUT / "bb_daily.parquet")
    print(json.dumps(out, indent=1, default=float))
