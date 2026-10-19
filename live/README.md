# live

`run_live_mix9_binance.py` trades MIX9_K15 on one Binance USDT-M sub-account, daily, with the ML01 paper engine's
signal code. Needs an ML01 checkout (`ML01_ROOT`, default `~/ML01`), its venv and the daily `panel_pit` refresh.

## Run

```bash
~/ML01/.venv/bin/python run_live_mix9_binance.py          # dry run: prints the orders, sends nothing
~/ML01/.venv/bin/python run_live_mix9_binance.py check    # live book vs paper_frozen MIX9_K15 book
MIX9_LIVE=1 ~/ML01/.venv/bin/python run_live_mix9_binance.py   # trades, unless LIVE_DRY_RUN=1 in .env.binance
```

| Setting | Where | Default |
|---------|-------|---------|
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | `.env.binance` next to the script | — |
| `LIVE_DRY_RUN` | `.env.binance`; master switch, 1 = never trade | 1 |
| `MIX9_LIVE` | launchd plist | 0 |
| `MIX9_NOTIONAL` (gross $) / `MIX9_LEVERAGE` | plist or env | 1000 / 3 |

## Behaviour

| Item | Rule |
|------|------|
| Schedule | 09:30 local (after ML01's 08:15 data refresh), retries 10:30 and 12:00; trades once per data bar |
| Data check | refuses to run unless the panel's last daily bar is yesterday (UTC) |
| Universe | basket coins in `TRADING` status on Binance now |
| Orders | market, diff vs current positions, reductions first, `reduceOnly` on closes, cross margin, one-way mode |
| Minimums | names below Binance's min notional are skipped and logged (BTC $50 at $33 per name) |
| Safety | refuses to trade if equity < 2 × margin; hedge mode → error |
| Records | `ledger_binance/orders.csv`, `account.csv`, `positions.csv`; `logs/live_binance.log`; Telegram alert on errors |
| Lag | `ag_*` / `ls_*` lagged 1 day (published a day late); paper uses them same-day |

## Parity (2026-10-19, bar 2026-10-18)

`check`: 200 / 200 basket names tradable; live vs paper overlap long 15 / 15, short 15 / 15. Dry run against a $800
account: 29 orders of ~$33, gross $956 (BTC short skipped by the minimum).

## Setup

1. Binance sub-account, one-way position mode, USDT in the USDT-M futures wallet (margin = notional / leverage).
2. API key: futures on, withdrawals off, IP whitelisted (IPv4; the script forces IPv4).
3. Copy `.env.example` to `.env.binance` next to the script, `chmod 600`, fill the keys.
4. Edit paths in `com.ml01.live-mix9-binance.plist`, copy to `~/Library/LaunchAgents/`, `launchctl load -w` it.
5. Stop: `LIVE_DRY_RUN=1` or `launchctl unload -w ~/Library/LaunchAgents/com.ml01.live-mix9-binance.plist`
   (neither closes open positions).

`run_live_mix9_okx.py`: the same trader for OKX, never run live (MIX9 fails on OKX's universe, see the main README).
