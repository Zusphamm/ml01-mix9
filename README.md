# ml01-mix9

MIX9_K15: a 9-alpha low-correlation mix of the ML01 grammar alphas, traded as one daily top-15 / bottom-15 book on
Binance USDT-M perps. Paper from 2026-10-20 (ML01 `paper_frozen` account `MIX9_K15`), live on a Binance sub-account
from the same day (`live/`).

## Construction

| Step | Rule |
|------|------|
| Universe | PIT top-200 Binance USDT-M perps by 30-day $ volume, delisted coins included (ML01 `panel_pit`) |
| Member score | `sign × winsor(cs_zscore(raw), 3)` per alpha, on the basket only |
| Composite | mean of the available member scores per coin |
| Book | 15 highest +0.5/15, 15 lowest −0.5/15 (equal weight, dollar neutral), signal at close d-1, daily |
| Costs | 5 bp taker + 2/5/10/20 bp tier by previous-day $ volume; real Binance funding |
| Reported gross | 1.27× (all Sharpe / return / DD below) |

Selection: the 39 `SINGLES_OS10` alphas (IS ≥ 1.4 and OS ≥ 1.0) clustered by average linkage on 1 − corr of IS
daily net (cut 0.5): 9 clusters, best IS Sharpe per cluster. Max pairwise |corr| inside the mix: 0.50.

## Members

| Sign | Key | IS Sharpe | Expression |
|------|-----|-----------|------------|
| +1 | `cond\|taker10\|taker20\|low\|s3` | 1.47 | `cs_zscore(ts_ema(cs_zscore(ts_mean(taker_buy_ratio, 10)) * (1 - rank(ts_mean(taker_buy_ratio, 20))), 3))` |
| −1 | `diverge\|fund_chg40\|taker20\|s5` | 1.85 | `cs_zscore(ts_ema(cs_zscore(ts_mean(funding, 3) - ts_mean(funding, 40)) - cs_zscore(ts_mean(taker_buy_ratio, 20)), 5))` |
| −1 | `diverge\|fund_chg60\|illiq10\|s5` | 2.58 | `cs_zscore(ts_ema(cs_zscore(ts_mean(funding, 3) - ts_mean(funding, 60)) - cs_zscore(log(ts_mean(abs(returns) / quote_volume, 10))), 5))` |
| −1 | `diverge\|size5\|taker5\|s5` | 2.97 | `cs_zscore(ts_ema(cs_zscore(log(ts_mean(quote_volume, 5))) - cs_zscore(ts_mean(taker_buy_ratio, 5)), 5))` |
| −1 | `diverge\|sweep20\|tk_us20\|s5` | 1.66 | `cs_zscore(ts_ema(cs_zscore(ts_mean(log(ag_sweep), 20)) - cs_zscore(ts_mean(tk_us, 20)), 5))` |
| −1 | `diverge\|trade_size10\|vwap_dev40\|s0` | 1.56 | `cs_zscore(cs_zscore(log(ts_mean(trade_size, 10))) - cs_zscore(ts_mean(log(close / vwap), 40)))` |
| +1 | `regime\|illiq20\|taker40\|dispersion\|s5` | 2.15 | `cs_zscore(ts_ema(where(st_dispersion > 0, cs_zscore(log(ts_mean(abs(returns) / quote_volume, 20))), cs_zscore(ts_mean(taker_buy_ratio, 40))), 5))` |
| +1 | `regime\|illiq60\|tk_us10\|mkt_vol\|s3` | 2.55 | `cs_zscore(ts_ema(where(st_mkt_vol > 0, cs_zscore(log(ts_mean(abs(returns) / quote_volume, 60))), cs_zscore(ts_mean(tk_us, 10))), 3))` |
| +1 | `regime\|vpin60\|smart_dumb10\|mkt_vol\|s5` | 1.55 | `cs_zscore(ts_ema(where(st_mkt_vol > 0, cs_zscore(ts_mean(ag_vpin, 60)), cs_zscore(log(ts_mean(ls_top_pos, 10)) - log(ts_mean(ls_global, 10)))), 5))` |

`grammar.py` decodes a key into its expression. Fields `ag_*` come from Binance aggTrades, `tk_us` is the taker-buy
share of the US session (1h klines), `ls_*` from Binance futures metrics; all built by ML01's data scripts.

## Results (MIX9 vs LIVE = ML01's single live alpha `cond|illiq60|dist_high60|low|s0`)

| Window | MIX9 Sharpe | MIX9 ret / yr | MIX9 max DD | LIVE Sharpe | LIVE max DD |
|--------|-------------|---------------|-------------|-------------|-------------|
| IS 2021-01 .. 2023-12 | 3.26 | +89.5% | −10.4% | 2.12 | −28.8% |
| OS 2024-01 .. 2026-06 (run once) | 2.46 | +100.8% | −20.3% | 1.81 | −33.3% |
| Blackbox 2026-07-01 .. 10-18 (109 d) | 1.43 | +72.6% | −22.0% | 0.72 | −15.4% |
| Last 30 days (09-19 .. 10-18) | −1.10 | −3.9% total | −10.4% | 1.38 | −4.9% |

IS DSR 1.00 (N 20), OS/IS 0.75. Blackbox by month: Jul +19.6%, Aug −24.0%, Sep +21.8%, Oct (to 10-18) +4.3%.
Correlation with LIVE: 0.29 IS, 0.25 OS, 0.20 blackbox. Daily series in `research/results/*_daily.csv`
(index = the UTC day the return is earned).

**Honest caveats**
- The 39 members were already picked with OS ≥ 1.0 and MIX9 is a post-hoc pick of 12 IS rows: OS and blackbox are
  not clean holdouts. Forward paper / live from 2026-10-20 is the real test.
- Volatile: Aug −24%, one −7% day (2026-09-24). Expect monthly swings of ±20% at 1.27× gross.

## Venue check: OKX (do not trade MIX9 there)

| Book | IS | OS | Blackbox |
|------|----|----|----------|
| MIX9 on Binance | 3.26 | 2.42 | 1.43 |
| MIX9, OKX listings only (Binance prices) | 2.09 | 1.73 | −1.52 |
| MIX9 on OKX (listings + prices + OKX volume cost) | 1.95 | 1.64 | −1.43 |

The edge lives in small coins listed only on Binance. `live/run_live_mix9_okx.py` is kept but was never run live.

## Files

| Path | Content |
|------|---------|
| `mix9.json` | members, signs, expressions, construction, all result tables |
| `grammar.py` | key → expression (copy of ML01 `alpha_lab/grammar.py`) |
| `research/mix_btc.py` | selection + IS of 12 mixes (prereg) · `mix_os.py` OS once · `mix_bb.py` blackbox · `okx_backtest.py` venue check |
| `research/results/` | prereg, IS / OS / blackbox / OKX JSON + daily CSVs |
| `research/notes-mix-btc.md` | ML01 research note (full log, incl. BTC filters that did not help) |
| `live/` | Binance live trader, launchd plist, setup ([live/README.md](live/README.md)) |

The research scripts and the live trader import ML01 (`alpha_lab`, `research/alpha_f_backtest`, `scripts/paper_frozen.py`)
and read its panels (`~/Desktop/datacryp/derived/panel_pit`); they do not run standalone.
