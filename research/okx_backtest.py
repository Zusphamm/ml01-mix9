"""MIX9 executed on OKX: OKX universe (point-in-time by first OKX candle), OKX prices, OKX $ volume for the cost tier
(user request 2026-10-19, before the OKX live account trades).

  python okx_backtest.py fetch    # OKX daily candles (1Dutc) of every live USDT-margined swap -> derived/okx/
  python okx_backtest.py run      # IS / OS / BB 2026-07-01 .. last bar; logged in out/os_access.log

Signal = the live trader's: MIX9 composite z from Binance panel_pit (members need Binance-only taker / aggTrades
fields), top-15 / bottom-15 among coins that have an OKX daily candle that day. Returns: OKX close-to-close of the next
day. Cost: 5bp taker + tier by trailing OKX $ volume (trial_screen.tier_cost_rate). Funding: Binance settlements as a
proxy (OKX's API keeps only ~3 months). Caveats: only swaps listed on OKX today are fetched (OKX's delisted swaps are
missing: survivorship); OS and BB were already read for MIX9 on Binance, so this is a venue check, not a holdout.
"""
import datetime as dt, json, math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
D = Path.home() / "Desktop/datacryp/derived/okx"
OUT_N = "okx_backtest"


def fetch():
    D.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    inst = s.get("https://www.okx.com/api/v5/public/instruments", params={"instType": "SWAP"}, timeout=30).json()["data"]
    inst = [i for i in inst if i["settleCcy"] == "USDT" and i["ctType"] == "linear"]
    json.dump(inst, open(D / "instruments.json", "w"))
    rows = {}
    for n, i in enumerate(inst):
        iid, after, got = i["instId"], "", []
        while True:
            for _ in range(5):
                try:
                    r = s.get("https://www.okx.com/api/v5/market/history-candles",
                              params={"instId": iid, "bar": "1Dutc", "limit": 100, **({"after": after} if after else {})}, timeout=30).json()
                    break
                except Exception:
                    time.sleep(3)
            data = r.get("data", [])
            if r.get("code") not in ("0", 0) or not data:
                if r.get("code") not in ("0", 0):
                    time.sleep(2)
                    if r.get("code") == "50011":
                        continue
                break
            got += data
            after = data[-1][0]
            time.sleep(0.12)
            if len(data) < 100:
                break
        if got:
            df = pd.DataFrame(got, columns=["ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"])
            df = df[df.confirm == "1"]
            df.index = pd.to_datetime(df.ts.astype(np.int64), unit="ms", utc=True)
            rows[iid] = df[["o", "c", "volCcyQuote"]].astype(float).sort_index()
        if n % 25 == 0:
            print(f"{n}/{len(inst)} {iid} {len(got)} bars", flush=True)
    close = pd.DataFrame({k: v.c for k, v in rows.items()}).sort_index()
    qv = pd.DataFrame({k: v.volCcyQuote for k, v in rows.items()}).sort_index()
    close.to_parquet(D / "close_1d.parquet"); qv.to_parquet(D / "quote_volume_1d.parquet")
    print("saved", close.shape, close.index.min(), close.index.max())


def okx_to_binance(cols, binance_cols):
    """OKX instId (BASE-USDT-SWAP) -> Binance perp symbol: BASEUSDT, else 1000BASEUSDT / 1000000BASEUSDT."""
    out = {}
    for c in cols:
        b = c.split("-")[0]
        for cand in (f"{b}USDT", f"1000{b}USDT", f"1000000{b}USDT", f"1M{b}USDT"):
            if cand in binance_cols:
                out[c] = cand
                break
    return out


def run():
    import trial_screen as S, live_options as LO, mix_btc as MB
    with open(S.OUT / "os_access.log", "a") as fh:
        fh.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()} okx_backtest.py IS+OS+BLACKBOX (until=None) venue check of MIX9 "
                 f"on OKX prices; MIX9 OS/BB already read on Binance; requested by user\n")
    out = S.OUT / OUT_N
    out.mkdir(parents=True, exist_ok=True)
    ks = json.loads((S.OUT / "mix_os/prereg.json").read_text())["members"]
    sign = MB.is_daily()[1]
    P = MB.load(None)
    close = P["close"]
    Z = {k: LO._z((k, sign[k]))[1] for k in sorted(set(ks) | {MB.LIVE})}
    comp = sum(Z[k].fillna(0) for k in ks) / sum(Z[k].notna().astype(int) for k in ks).replace(0, np.nan)
    oc, oq = pd.read_parquet(D / "close_1d.parquet"), pd.read_parquet(D / "quote_volume_1d.parquet")
    m = okx_to_binance(oc.columns, set(close.columns))
    oc = oc[list(m)].rename(columns=m).T.groupby(level=0).first().T.reindex(index=close.index, columns=close.columns)
    oq = oq[list(m)].rename(columns=m).T.groupby(level=0).first().T.reindex(index=close.index, columns=close.columns)
    on_okx = oc.notna()
    fwd_okx = oc.pct_change(fill_method=None).shift(-1).where(oc.shift(-1).notna()).fillna(0).clip(-0.95, 3)
    rate_okx = S.tier_cost_rate(oq.shift(1)).reindex_like(close)

    def pnl(W, fwd, rate):
        W = W.reindex_like(close).fillna(0)
        held, dW = W.shift(1).fillna(0), (W - W.shift(1)).abs()
        net = (held * fwd).sum(1) - (held * S.G["fund"].shift(-1).fillna(0)).sum(1) - (dW * rate.fillna(0.0025)).sum(1)
        return net.iloc[:-1] * MB.LEV

    books = {
        "LIVE (Binance)": pnl(LO.topk(Z[MB.LIVE], 15), S.G["fwd"], S.G["cost_rate"]),
        "MIX9 Binance": pnl(LO.topk(comp, 15), S.G["fwd"], S.G["cost_rate"]),
        "MIX9 OKX universe, Binance prices": pnl(LO.topk(comp.where(on_okx), 15), S.G["fwd"], S.G["cost_rate"]),
        "MIX9 on OKX (universe + prices + OKX volume cost)": pnl(LO.topk(comp.where(on_okx), 15), fwd_okx, rate_okx),
    }
    f = MB.smb(P)
    segs = {"IS 2021-23": (S.IS0, S.OS0), "OS 2024-26H1": (S.OS0, S.BLACKBOX0), "BB 07-01..": (S.BLACKBOX0, pd.Timestamp("2100-01-01", tz="UTC"))}
    res = {}
    for b, r in books.items():
        for sg, (lo, hi) in segs.items():
            x = r[(r.index >= lo) & (r.index < hi)]
            res[f"{b} | {sg}"] = MB.stats(x, f, books["LIVE (Binance)"].reindex(x.index))
    n_okx = on_okx.where(comp.notna()).sum(1)
    res["okx_names_in_basket"] = {str(y): int(n_okx[n_okx.index.year == y].median()) for y in range(2021, 2027)}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    pd.DataFrame(books).to_parquet(out / "daily.parquet")
    for k, v in res.items():
        print(k, json.dumps(v, default=float))


if __name__ == "__main__":
    {"fetch": fetch, "run": run}[sys.argv[1]]()
