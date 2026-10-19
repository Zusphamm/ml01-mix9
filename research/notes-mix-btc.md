# mix-btc

Low-correlation mix of the 39 SINGLES_OS10 alphas with a small BTC filter, as one live-tradable top-K book (user request
2026-10-19). Scripts `mix_btc.py` (selection on IS → prereg → IS) and `mix_os.py` (one confirmatory OS), outputs
`out/mix_btc/`, `out/mix_os/`. Engine = the live trader's (`live_options`: composite z, top-K, tiered cost, real
funding), 1.27× gross.

## Selection (IS daily returns only)

| Set | Rule | Members | Max pair corr |
|-----|------|---------|---------------|
| CLUST | best IS Sharpe in each of 9 correlation clusters (average linkage, corr ≥ 0.5 merges) | 9 | 0.50 |
| GREEDY | down the IS Sharpe ranking, keep if \|corr\| ≤ 0.4 with all kept, max 8 | 6 | 0.38 |

CLUST = taker10/taker20, fund_chg40/taker20, fund_chg60/illiq10, size5/taker5, sweep20/tk_us20, trade_size10/vwap_dev40,
illiq20/taker40 regime, illiq60/tk_us10 regime, vpin60/smart_dumb10 regime. The live alpha is in neither set.

## IS 2021-01 .. 2023-12 (12 trials, primary CLUST_K15_trend)

| Book | Sharpe | Return / yr | Max DD | Worst 30 d | corr small-minus-large | corr LIVE |
|------|--------|-------------|--------|------------|------------------------|-----------|
| LIVE | 2.12 | +60% | −28.8% | −21.0% | 0.52 | 1 |
| CLUST K15 none / trend / hedge | 3.26 / 3.09 / 3.19 | +90 / +72 / +89% | −10.4 / −10.2 / −10.4% | −9.2% | 0.07 | 0.29 |
| CLUST K30 none / trend / hedge | 3.09 / 3.00 / 2.92 | +65 / +52 / +63% | −9.2 / −9.7 / −10.7% | −7.9% | 0.16 | 0.36 |
| GREEDY K15 none / trend / hedge | 3.37 / 3.35 / 3.15 | +103 / +83 / +98% | −12.6 / −9.8 / −10.8% | −9.7% | 0.00 | 0.27 |
| GREEDY K30 none / trend / hedge | 2.59 / 2.57 / 2.48 | +55 / +45 / +54% | −11.1 / −11.1 / −12.6% | −10.3% | 0.02 | 0.33 |

Primary failed only its "filter earns its place" check (trend 3.09 < none 3.26) → stop per prereg. **The BTC filters
add nothing**: trend-50d halving cuts return more than risk in every set; the beta hedge is neutral (books are ≈ 0 beta).

## Confirmatory OS 2024-01 .. 2026-06 (user choice: CLUST_K15_none, post-hoc pick of 12, run once)

| Book | Sharpe | Return / yr | Max DD | Worst 30 d | P(30 d < 0) | 2024 / 2025 / 2026 H1 | corr LIVE |
|------|--------|-------------|--------|------------|-------------|-----------------------|-----------|
| LIVE | 1.81 | +61% | −33.3% | −26.5% | 28% | 1.66 / 1.48 / 2.79 | 1 |
| **MIX9 (CLUST_K15_none)** | **2.46** | +101% | **−20.3%** | **−15.5%** | 27% | 1.79 / 1.75 / 4.50 | 0.25 |

WFE 0.75; passes all three checks (WFE ≥ 0.6, Sharpe ≥ LIVE, DD better). **Not a clean holdout**: the 39 components
were picked with OS ≥ 1.0 and the variant was chosen after IS. BLACKBOX was not used.

## Blackbox 2026-07-01 .. 10-17 (mix_bb.py, until=None, run once, 109 days)

| Book | Sharpe | Return / yr | Max DD | Worst 30 d | Jul / Aug / Sep / Oct 1–17 | corr LIVE |
|------|--------|-------------|--------|------------|----------------------------|-----------|
| LIVE | 0.72 | +33% | −15.4% | −10.4% | +3.2 / +10.1 / −6.1 / +2.7% | 1 |
| MIX9 | **1.43** | +73% | **−22.0%** | **−23.8%** | +19.6 / **−24.0** / +21.8 / +4.3% | 0.20 |
| 50/50 LIVE + MIX9 (post hoc) | 1.41 | | −10.7% | | | |

Checks: Sharpe > 0 yes, max DD better than LIVE **no**. MIX9 doubled LIVE's blackbox Sharpe but swung hard
(August −24%, worst day −5.7% on 08-18). The two books correlate 0.20, so the 50/50 split keeps the Sharpe and halves
the drawdown. The 50/50 track is available from paper without a new account: average of LIVE15 and MIX9_K15.

## On OKX data (okx_backtest.py, 2026-10-19)

494 OKX USDT swaps, daily 1Dutc candles from listing (`derived/okx/`); universe point-in-time by first OKX candle
(48 basket names on OKX in 2021, 128 in 2026), OKX close-to-close returns, cost tier on OKX $ volume, Binance funding
as proxy. OKX's delisted swaps are missing (survivorship).

| Book | IS 2021–23 | OS 2024–26 H1 (2024 / 2025 / 2026) | BB 07-01 .. 10-17 | OS max DD |
|------|------------|------------------------------------|-------------------|-----------|
| LIVE (Binance) | 2.11 | 1.90 (1.66 / 1.48 / 3.11) | 0.72 | −33.3% |
| MIX9 Binance | 3.26 | 2.42 (1.79 / 1.75 / 4.31) | 1.43 | −20.3% |
| MIX9, OKX universe, Binance prices | 2.09 | 1.73 (0.32 / 1.90 / 3.35) | −1.52 | −21.6% |
| **MIX9 on OKX** (universe + prices + cost) | **1.95** | **1.64** (0.22 / 1.77 / 3.28) | **−1.43** (−44% / yr) | −22.7% |

Restricting to coins listed on OKX at the time removes about a third of MIX9's IS Sharpe and turns the blackbox
negative; OKX prices and costs cost little on top. MIX9-on-OKX is below LIVE in IS and OS and loses in the blackbox:
**the OKX live launch is not supported by this test.**

## Next

- Paper: `MIX9_K15` added to `paper_frozen/BLEND_LIVE_config.json` (same accounts / fills as LIVE15, first fill
  2026-10-20, no backfill). The forward paper is the clean test before moving the live account.
- **Binance sub-account instead (2026-10-19):** `system-mix9/run_live_mix9_binance.py`, keys in `.env.binance` ($800 USDT-M,
  futures only, no withdrawals), launchd `com.ml01.live-mix9-binance` 09:30 / 10:30 / 12:00 local, $1,000 gross, 3x cross,
  one-way; trades only when `LIVE_DRY_RUN=0` in `.env.binance` (set to 0 on 2026-10-19 by the user: first live trade 2026-10-20 09:30 local, same day as paper). Bar 2026-10-18 book = paper
  MIX9_K15 15/15 long and short; BTC short skipped (Binance min $50 > $33 per name). Ledger `system-mix9/ledger_binance/`.
- **OKX live paused 2026-10-19 before the first trade** (user, after the OKX-data backtest below; launchd job unloaded, no order sent). Was: `~/Desktop/alpha-live/system-mix9/run_live_mix9_okx.py`, launchd
  `com.ml01.live-mix9-okx` 09:30 / 10:30 / 12:00 local, $1,000 gross, 3x cross, net mode; ledger in `system-mix9/ledger/`.
  Only 127 of the 200 basket coins trade on OKX, so ranking is restricted to them: the 2026-10-19 book shares 7 / 15
  longs and 13 / 15 shorts with paper. IS with today's OKX list (survivorship caveat): Sharpe 2.67, max DD −12.8%
  (full basket 3.26 / −10.4%; LIVE 2.12 / −28.8%).
- Live: members read session taker (`tk_us*`) and aggTrades (`sweep`, `vpin`) fields, so the live trader needs the
  09:20 panel path with ag_* lagged 1 day (as `~/ml01-top8-alphas/live/run_live.py`), not the 07:00 REST path.

## Related

- [live-plus4.md](live-plus4.md) · [size-neutral.md](size-neutral.md) · [blend-live.md](blend-live.md) · [live-options.md](live-options.md)
