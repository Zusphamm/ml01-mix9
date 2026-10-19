"""
Live trader: MIX9 composite book on a Binance USDT-M sub-account (one-way mode, cross margin), user request 2026-10-19
(after the OKX backtest: MIX9 on OKX's smaller universe lost its edge, notes/mix-btc.md).
Book   : the paper account MIX9_K15 of ~/ML01/paper_frozen/BLEND_LIVE_config.json: composite z = mean over 9 low-correlation
         SINGLES_OS10 alphas of sign x winsor(zscore, 3) on the PIT top-200 basket; 15 highest long, 15 lowest short,
         equal weight. Research: ~/ML01/research/alpha_f_backtest/notes/mix-btc.md (IS 3.26, OS 2.46, BB 1.43; OS / BB
         not clean holdouts).
Live differences vs paper (logged every run): (1) ranking restricted to coins in TRADING status on Binance now;
         (2) ag_* / ls_* fields lagged 1 day (published a day late; blackbox variant B), so all 9 members are present at
         run time; (3) fills at the run time (after the 08:15 local data refresh), not 01:00 UTC.
Data   : panel path only (members need session taker / aggTrades fields): bar d-1 must be in panel_pit; run 09:30 local,
         retries 10:30 and 12:00. Idempotent per data bar; orders are diffs vs the account.
Sizing : fixed gross notional (MIX9_NOTIONAL, default 1000 USDT) -> notional/30 per name; leverage MIX9_LEVERAGE (3).
         Names below Binance's min notional (BTC $100 etc.) are skipped and logged.

  ~/ML01/.venv/bin/python run_live_mix9_binance.py           # DRY RUN (default): prints orders, sends nothing
  MIX9_LIVE=1 ~/ML01/.venv/bin/python run_live_mix9_binance.py   # sends market orders, unless LIVE_DRY_RUN=1 in .env.binance
  ~/ML01/.venv/bin/python run_live_mix9_binance.py check     # book vs the paper engine's MIX9_K15 book
Env (.env.binance here): BINANCE_API_KEY / BINANCE_API_SECRET (sub-account, IPv4 whitelisted), LIVE_DRY_RUN (master
switch: 1 = never trade), MIX9_NOTIONAL, MIX9_LEVERAGE.
"""
import socket
_gai = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [r for r in _gai(*a, **k) if r[0] == socket.AF_INET]   # key whitelists IPv4 only

import json, logging, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ML01 = Path(os.getenv("ML01_ROOT", Path.home() / "ML01")).expanduser()
RES = ML01 / "research/alpha_f_backtest"
CFG = ML01 / "paper_frozen/BLEND_LIVE_config.json"
load_dotenv(HERE / ".env.binance")
KEY, K = "MIX9_K15", 15
NOTIONAL = float(os.getenv("MIX9_NOTIONAL", "1000"))
LEVERAGE = int(os.getenv("MIX9_LEVERAGE", "3"))
LIVE = os.getenv("MIX9_LIVE", "0") == "1" and os.getenv("LIVE_DRY_RUN", "1") != "1"
STATE, LEDGER = HERE / "state_binance.json", HERE / "ledger_binance"
LAG = ("ag_", "ls_")
EXTRA = ("taker_buy_quote_volume", "count", "oi_value", "ls_top_acct", "ls_top_pos", "ls_global", "taker_ls", "spot_close",   # = paper_frozen._grammar_env
         "prem_close", "prem_high", "prem_low", "i_rv", "i_rskew", "i_rkurt", "i_downvar", "i_jump", "i_ac1", "i_roll",
         "r_asia", "r_eu", "r_us", "vs_asia", "vs_eu", "vs_us", "tk_asia", "tk_eu", "tk_us", "tk_late",
         "ag_lg", "ag_lgimb", "ag_smimb", "ag_vpin", "ag_lam", "ag_sweep", "ag_burst")
PIT = Path.home() / "Desktop/datacryp/derived/panel_pit/1d"

(HERE / "logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [mix9-bn] %(message)s",
                    handlers=[logging.FileHandler(HERE / "logs/live_binance.log"), logging.StreamHandler()])
log = logging.getLogger("mix9")


def retry(fn, *a, tries=5, wait=20, **k):
    for i in range(tries):
        try:
            return fn(*a, **k)
        except (ccxt.DDoSProtection, ccxt.RateLimitExceeded, ccxt.NetworkError) as e:
            if i == tries - 1:
                raise
            log.warning(f"  {type(e).__name__}: retry in {wait}s ({str(e)[:80]})")
            time.sleep(wait)


# --------------------------------------------------------------------------- signal (ML01 paper engine, ag/ls lagged)
def _ml01():
    os.chdir(RES)
    for p in (str(ML01 / "scripts"), str(ML01), str(RES)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import live6m as L
    return L


def env_today(L, d):
    """paper_frozen._grammar_env for day d (panels truncated at close d-1), with ag_* / ls_* shifted 1 day."""
    import repo_pit as R
    from alpha_lab.dsl import build_env
    s = d - pd.Timedelta(days=1)
    P = {k: pd.read_parquet(R.PD / f"d_{k}.parquet").astype(float).reindex_like(L.close).loc[:s] for k in ("open", "high", "low")}
    P.update(close=L.close.loc[:s], volume=L.vol.loc[:s], quote_volume=L.qv.loc[:s])
    for f in EXTRA:
        if (PIT / f"{f}.parquet").exists():
            x = pd.read_parquet(PIT / f"{f}.parquet").astype(float).reindex_like(L.close).loc[:s]
            P[f] = x.shift(1) if f.startswith(LAG) else x
    env = build_env(P)
    mem, _ = L.basket(L.close.loc[:s], L.qv.loc[:s], d, 30)
    mask = pd.Series(False, index=L.close.columns)
    mask[list(mem & set(L.close.columns))] = True
    return env, mask & L.close.loc[:s].iloc[-1].notna()


def composite(a, env, mask):
    """Mean member z per coin (paper_frozen._blend_weights before the top-K step)."""
    from alpha_lab import grammar as G
    from alpha_lab.dsl import evaluate
    from alpha_lab.operators import cs_winsorize, cs_zscore
    comp = 0
    for g in a["blend"]:
        zs = []
        for key, sign in g["keys"]:
            raw = evaluate(G.expr(G.parse_key(key)), env).iloc[-1:].replace([np.inf, -np.inf], np.nan)
            zs.append((sign * cs_winsorize(cs_zscore(raw.where(mask.to_frame().T.values)), 3)).iloc[-1])
        comp = comp + g["w"] * pd.concat(zs, axis=1).mean(axis=1)
    return comp.dropna()


def book(z, tradable):
    z = z[z.index.isin(tradable)]
    if len(z) < 2 * K:
        raise RuntimeError(f"only {len(z)} tradable names with a signal")
    w = {s: 0.5 / K for s in z.nlargest(K).index}
    w.update({s: -0.5 / K for s in z.nsmallest(K).index})
    return w


def signal():
    L = _ml01()
    today = pd.Timestamp.now(tz="UTC").normalize()
    bar = L.close.index.max()
    if bar + pd.Timedelta(days=1) != today:
        raise RuntimeError(f"daily panel last bar {bar.date()} != yesterday: data refresh not done yet")
    a = [x for x in json.loads(CFG.read_text())["alphas"] if x["key"] == KEY][0]
    env, mask = env_today(L, today)
    return composite(a, env, mask), str(bar.date()), a, L


# --------------------------------------------------------------------------- Binance USDT-M
def exchange(auth=True):
    k = dict(apiKey=os.getenv("BINANCE_API_KEY", ""), secret=os.getenv("BINANCE_API_SECRET", "")) if auth else {}
    ex = ccxt.binanceusdm({**k, "enableRateLimit": True})
    retry(ex.load_markets)
    return ex


def bn_map(ex, names):
    """panel_pit symbol (= Binance id) -> ccxt symbol, only linear USDT perps in TRADING status."""
    out = {}
    for m in ex.markets.values():
        if m.get("swap") and m.get("linear") and m.get("settle") == "USDT" and m["info"].get("contractType") == "PERPETUAL" \
                and m["info"].get("status") == "TRADING" and m["id"] in names:
            out[m["id"]] = m["symbol"]
    return out


def account(ex):
    if ex.fapiPrivateGetPositionSideDual()["dualSidePosition"]:
        raise RuntimeError("Binance account is in hedge mode: switch to one-way mode")
    a = ex.fapiPrivateV2GetAccount()
    pos = {ex.markets_by_id[p["symbol"]][0]["symbol"]: float(p["positionAmt"]) for p in a["positions"] if float(p["positionAmt"])}
    return float(a["totalMarginBalance"]), pos


def _qty(ex, s, x):
    try:
        return float(ex.amount_to_precision(s, x))
    except ccxt.InvalidOrder:
        return 0.0


def _min_cost(m):
    return (m["limits"]["cost"] or {}).get("min") or 5.0


def plan(ex, w, smap, pos, prices):
    """Market orders to move pos -> target. Closes / reductions first (they free margin), then opens / adds."""
    tgt = {}
    for s, wt in w.items():
        m, px = ex.market(smap[s]), prices[smap[s]]
        q = _qty(ex, m["symbol"], NOTIONAL * abs(wt) / px)
        if q * px < _min_cost(m) or q < ((m["limits"]["amount"] or {}).get("min") or 0):
            log.warning(f"    skip {s}: ${NOTIONAL * abs(wt):.1f} below the exchange minimum (${_min_cost(m):.0f} / step)")
            continue
        tgt[m["symbol"]] = q if wt > 0 else -q
    orders = []
    for s in set(tgt) | set(pos):
        cur, new = pos.get(s, 0.0), tgt.get(s, 0.0)
        d = new - cur
        if d == 0:
            continue
        reduce = abs(new) < abs(cur) and np.sign(new) in (0, np.sign(cur))
        if reduce or new == 0:
            q = _qty(ex, s, abs(d))
            if q > 0:
                orders.append(dict(symbol=s, side="buy" if d > 0 else "sell", qty=q, reduce=True, usd=q * prices.get(s, 0)))
        else:
            if cur != 0 and np.sign(new) != np.sign(cur):                 # flip: close first, then open
                orders.append(dict(symbol=s, side="buy" if cur < 0 else "sell", qty=abs(cur), reduce=True, usd=abs(cur) * prices.get(s, 0)))
                d = new
            q = _qty(ex, s, abs(d))
            if q * prices[s] >= _min_cost(ex.market(s)):
                orders.append(dict(symbol=s, side="buy" if d > 0 else "sell", qty=q, reduce=False, usd=q * prices[s]))
    return sorted(orders, key=lambda o: not o["reduce"]), tgt


def _append(name, rows, key):
    if not rows:
        return
    LEDGER.mkdir(exist_ok=True)
    f = LEDGER / f"{name}.csv"
    seen = set(pd.read_csv(f, usecols=[key], dtype=str)[key]) if f.exists() else set()
    new = [r for r in rows if str(r[key]) not in seen]
    if new:
        pd.DataFrame(new).to_csv(f, mode="a", header=not f.exists(), index=False)


def snapshot(ex, bar):
    a = ex.fapiPrivateV2GetAccount()
    t = ex.milliseconds()
    P = [p for p in a["positions"] if float(p["positionAmt"])]
    _append("account", [dict(snap_id=str(t), utc=ex.iso8601(t), data_bar=bar, total_eq=float(a["totalMarginBalance"]),
                             wallet=float(a["totalWalletBalance"]), n_pos=len(P), upnl=float(a["totalUnrealizedProfit"]))], "snap_id")
    _append("positions", [dict(row_id=f"{t}-{p['symbol']}", utc=ex.iso8601(t), symbol=p["symbol"], amount=float(p["positionAmt"]),
                               notional=float(p.get("notional") or 0), upnl=float(p.get("unrealizedProfit") or 0)) for p in P], "row_id")
    log.info(f"  ledger: equity ${float(a['totalMarginBalance']):.2f}, {len(P)} positions")


def alert(text):
    try:
        sys.path.insert(0, str(Path.home() / "Desktop/alpha-live/system"))
        import tele_report
        tele_report.alert(f"[MIX9 Binance] {text}")
    except Exception as e:
        log.error(f"  telegram alert failed: {str(e)[:120]}")


def check():
    """Live book (Binance TRADING names) vs the paper engine's MIX9_K15 book for the same day."""
    z, bar, a, L = signal()
    import paper_frozen as PF
    today = pd.Timestamp(bar, tz="UTC") + pd.Timedelta(days=1)
    env, mask, _ = PF._grammar_env(L, today)
    pw = PF._blend_weights(a, env, mask); pw = pw[pw != 0]
    ex = exchange(auth=False)
    smap = bn_map(ex, set(z.index))
    w = book(z, smap)
    ov_l = len(set(k for k, v in w.items() if v > 0) & set(pw[pw > 0].index))
    ov_s = len(set(k for k, v in w.items() if v < 0) & set(pw[pw < 0].index))
    log.info(f"check bar {bar}: tradable {len(smap)}/{len(z)} names with a signal; live vs paper overlap long {ov_l}/15, short {ov_s}/15")
    log.info(f"  paper names not tradable now: {sorted(set(pw.index) - set(smap))}")


def main():
    log.info("=" * 60)
    log.info(f"{KEY} top {K}/{K} on Binance | notional ${NOTIONAL:.0f} | lev {LEVERAGE}x cross | {'LIVE' if LIVE else 'DRY RUN'}")
    if sys.argv[1:] == ["check"]:
        return check()
    z, bar, _, _ = signal()
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    if LIVE and st.get("data_bar") == bar:
        log.info(f"  bar {bar} already traded at {st.get('at')}")
        return
    ex = exchange(auth=True)
    smap = bn_map(ex, set(z.index))
    w = book(z, smap)
    log.info(f"  signal bar {bar}: {len(smap)}/{len(z)} names tradable on Binance")
    log.info(f"  long  {sorted(s for s, x in w.items() if x > 0)}")
    log.info(f"  short {sorted(s for s, x in w.items() if x < 0)}")
    try:
        equity, pos = retry(account, ex)
    except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
        if LIVE:
            raise
        log.warning(f"  account not readable ({str(e)[:140]}); DRY RUN assumes a flat account")
        equity, pos = float("nan"), {}
    syms = sorted({smap[s] for s in w} | set(pos))
    prices = {s: float(t["last"]) for s, t in retry(ex.fetch_tickers).items() if s in syms}
    orders, tgt = plan(ex, {s: x for s, x in w.items()}, smap, pos, prices)
    gross = sum(abs(q) * prices[s] for s, q in tgt.items())
    log.info(f"  equity ${equity:.2f} | target {len(tgt)} names gross ${gross:.0f} | {len(orders)} orders")
    for o in orders:
        log.info(f"    {'[DRY] ' if not LIVE else ''}{o['side']:4s} {o['qty']} {o['symbol']} (${o['usd']:.1f}){' reduceOnly' if o['reduce'] else ''}")
    if not LIVE:
        return
    if equity < 2 * NOTIONAL / LEVERAGE:
        raise RuntimeError(f"equity ${equity:.0f} < 2 x margin ${NOTIONAL / LEVERAGE:.0f}: refusing to trade")
    ok, sent = 0, []
    for o in orders:
        try:
            if not o["reduce"]:
                try: ex.set_margin_mode("cross", o["symbol"])
                except Exception as e:
                    if "No need to change" not in str(e): log.warning(f"    margin mode {o['symbol']}: {str(e)[:80]}")
                try: ex.set_leverage(LEVERAGE, o["symbol"])
                except Exception as e: log.warning(f"    leverage {o['symbol']}: {str(e)[:80]}")
            r = ex.create_order(o["symbol"], "market", o["side"], o["qty"],
                                params={**({"reduceOnly": True} if o["reduce"] else {}), "newOrderRespType": "RESULT"})
            ok += 1; time.sleep(0.2)
            sent.append(dict(order_id=r["id"], utc=ex.iso8601(ex.milliseconds()), data_bar=bar, symbol=o["symbol"], side=o["side"],
                             qty=o["qty"], reduce_only=o["reduce"], usd=round(o["usd"], 2), error=""))
        except Exception as e:
            log.error(f"    FAIL {o['symbol']}: {str(e)[:160]}")
            sent.append(dict(order_id=f"fail-{ex.milliseconds()}-{o['symbol']}", utc=ex.iso8601(ex.milliseconds()), data_bar=bar,
                             symbol=o["symbol"], side=o["side"], qty=o["qty"], reduce_only=o["reduce"], usd=round(o["usd"], 2), error=str(e)[:300]))
    log.info(f"  executed {ok}/{len(orders)}")
    _append("orders", sent, "order_id")
    try:
        snapshot(ex, bar)
    except Exception as e:
        log.error(f"  ledger snapshot failed: {str(e)[:160]}")
    if ok < len(orders):
        alert(f"bar {bar}: khớp {ok}/{len(orders)} lệnh; lần chạy kế tiếp sẽ đặt lại phần còn thiếu.")
        return
    STATE.write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(), data_bar=bar, notional=NOTIONAL, equity=equity,
                                     executed=ok, orders=len(orders)), indent=1))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if LIVE:
            msg = f"Lỗi: <code>{str(e)[:300]}</code>"
            if "-2015" in str(e) or "-2014" in str(e):
                try: msg += f"\nIP hiện tại: <b>{__import__('requests').get('https://api.ipify.org', timeout=5).text}</b> → thêm vào Binance API whitelist"
                except Exception: pass
            alert(msg)
        raise
