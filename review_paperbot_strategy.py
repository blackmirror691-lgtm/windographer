# -*- coding: utf-8 -*-

import re
import csv
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import bisect
import json
import time
import urllib.parse
import urllib.request
import urllib.error
from decimal import Decimal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from numba import jit

UTC = dt.timezone.utc
EPS = 1e-12

SNAP_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})Z .*?'
    r'(?P<slug>btc-updown-15m-(?P<start_epoch>\d+)).*?'
    r'Up\[(?P<up_bid>\d+(?:\.\d+)?)\([^]]*?\)/(?P<up_ask>\d+(?:\.\d+)?)\([^]]*?\)\]\s*'
    r'Dn\[(?P<dn_bid>\d+(?:\.\d+)?)\([^]]*?\)/(?P<dn_ask>\d+(?:\.\d+)?)\([^]]*?\)\]'
)
EXTHIST_RE = re.compile(r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})Z .*?\[ExtHist\]\s+.*?last=(?P<px>\d+\.?\d*)')
STRIKE_RE = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})Z .*?\[Strike\]\s+'
    r'(?P<slug>btc-updown-15m-(?P<start_epoch>\d+))\s+strike\([^)]+\)=(?P<strike>\d+\.?\d*)'
)


@dataclass
class Snap:
    t: int
    up_bid: float
    up_ask: float
    dn_bid: float
    dn_ask: float


class ExtSeries:
    def __init__(self):
        self.t: List[int] = []
        self.px: List[float] = []

    def add(self, epoch: int, px: float):
        if self.t and epoch < self.t[-1]:
            i = bisect.bisect_left(self.t, epoch)
            self.t.insert(i, epoch)
            self.px.insert(i, px)
        else:
            self.t.append(epoch)
            self.px.append(px)

    def nearest(self, epoch: int, max_diff: float = 120.0) -> Optional[float]:
        if not self.t:
            return None
        i = bisect.bisect_left(self.t, epoch)
        cand = []
        if i < len(self.t):
            cand.append(i)
        if i > 0:
            cand.append(i - 1)
        best = None
        best_diff = None
        for j in cand:
            diff = abs(self.t[j] - epoch)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best = self.px[j]
        if best_diff is not None and best_diff <= max_diff:
            return best
        return None


@jit(nopython=True, cache=True)
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@jit(nopython=True, cache=True)
def lerp(a: float, b: float, w: float) -> float:
    return a + (b - a) * w


@jit(nopython=True, cache=True)
def intersect_interval(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    if a[0] > a[1] or b[0] > b[1]:
        return (2.0, 1.0)
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo <= hi + EPS:
        return (lo, hi)
    return (2.0, 1.0)


@jit(nopython=True, cache=True)
def interval_for_linear_between(v0: float, v1: float, lo: float, hi: float) -> Tuple[float, float]:
    dv = v1 - v0
    if abs(dv) < 1e-15:
        if (lo - EPS) <= v0 <= (hi + EPS):
            return (0.0, 1.0)
        return (2.0, 1.0)
    w1 = (lo - v0) / dv
    w2 = (hi - v0) / dv
    w_lo = clamp(min(w1, w2), 0.0, 1.0)
    w_hi = clamp(max(w1, w2), 0.0, 1.0)
    if w_lo <= w_hi + EPS:
        return (w_lo, w_hi)
    return (2.0, 1.0)


@jit(nopython=True, cache=True)
def interval_for_linear_leq(v0: float, v1: float, thr: float) -> Tuple[float, float]:
    dv = v1 - v0
    if abs(dv) < 1e-15:
        return (0.0, 1.0) if v0 <= thr + EPS else (2.0, 1.0)
    w_cross = (thr - v0) / dv
    if dv > 0:
        w_lo, w_hi = 0.0, w_cross
    else:
        w_lo, w_hi = w_cross, 1.0
    w_lo = clamp(w_lo, 0.0, 1.0)
    w_hi = clamp(w_hi, 0.0, 1.0)
    if w_lo <= w_hi + EPS:
        return (w_lo, w_hi)
    return (2.0, 1.0)


@jit(nopython=True, cache=True)
def segment_first_entry_w(t0: int, t1: int, ask0: float, ask1: float, bid0: float, bid1: float,
                          t_min: float, lb_ask: float, ub_ask: float, spread_max: float):
    if t1 <= t0:
        if (lb_ask - EPS) <= ask0 <= (ub_ask + EPS) and (ask0 - bid0) <= spread_max + EPS and float(t0) >= t_min - EPS:
            return float(t0), float(ask0), float(bid0)
        return None
    w_min = clamp((t_min - float(t0)) / float(t1 - t0), 0.0, 1.0)
    I = intersect_interval((w_min, 1.0), interval_for_linear_between(ask0, ask1, lb_ask, ub_ask))
    if I[0] > I[1]:
        return None
    I = intersect_interval(I, interval_for_linear_leq(ask0 - bid0, ask1 - bid1, spread_max))
    if I[0] > I[1]:
        return None
    w = I[0]
    ask_t = lerp(ask0, ask1, w)
    bid_t = lerp(bid0, bid1, w)
    t_entry = float(t0) + float(t1 - t0) * w
    if not ((lb_ask - EPS) <= ask_t <= (ub_ask + EPS)):
        return None
    if (ask_t - bid_t) > spread_max + EPS:
        return None
    if t_entry < t_min - EPS:
        return None
    return t_entry, ask_t, bid_t


@jit(nopython=True, cache=True)
def segment_first_stop_w(t0: int, t1: int, trigger_px0: float, trigger_px1: float,
                         fill_px0: float, fill_px1: float, t_min: float, stop_price: float):
    if t1 <= t0:
        if trigger_px0 <= stop_price + EPS and float(t0) >= t_min - EPS:
            return float(t0), float(trigger_px0), float(fill_px0)
        return None
    w_min = clamp((t_min - float(t0)) / float(t1 - t0), 0.0, 1.0)
    I = intersect_interval((w_min, 1.0), interval_for_linear_leq(trigger_px0, trigger_px1, stop_price))
    if I[0] > I[1]:
        return None
    w = I[0]
    trigger_t = lerp(trigger_px0, trigger_px1, w)
    fill_t = lerp(fill_px0, fill_px1, w)
    t_stop = float(t0) + float(t1 - t0) * w
    if trigger_t > stop_price + EPS or t_stop < t_min - EPS:
        return None
    return t_stop, trigger_t, fill_t


def simulate_one_market_settle(seq: List[Snap], start_epoch: int, strike_map: Dict[str, float], ext: ExtSeries, *, slug: str,
                               interval_sec: int, stake_usdc: float, winner_map: Dict[str, str], entry_price_max: float,
                               entry_floor: float, last_window_sec: int, stop_price: float, stop_slip: float,
                               min_prob: float, max_prob: float, spread_max: float, tie_wins: str) -> dict:
    close_epoch = start_epoch + int(interval_sec)
    if not seq or len(seq) < 2:
        return {"slug": slug, "traded": 0, "reason": "no_data"}

    strike = strike_map.get(slug)
    if strike is None:
        strike = ext.nearest(start_epoch, max_diff=300.0)

    t_window_start = float(close_epoch - int(last_window_sec))
    lb_ask = max(float(entry_floor), float(min_prob))
    ub_ask = min(float(entry_price_max), float(max_prob))
    if lb_ask > ub_ask + EPS:
        return {"slug": slug, "traded": 0, "reason": "no_trade"}

    entry_side = None
    entry_t = None
    entry_fill = None
    entry_idx_right = 1

    for i in range(1, len(seq)):
        s0 = seq[i - 1]
        s1 = seq[i]
        if s1.t < start_epoch:
            continue
        if s0.t > close_epoch:
            break
        up = segment_first_entry_w(s0.t, s1.t, s0.up_ask, s1.up_ask, s0.up_bid, s1.up_bid,
                                   t_window_start, lb_ask, ub_ask, float(spread_max))
        dn = segment_first_entry_w(s0.t, s1.t, s0.dn_ask, s1.dn_ask, s0.dn_bid, s1.dn_bid,
                                   t_window_start, lb_ask, ub_ask, float(spread_max))
        candidates = []
        if up is not None:
            candidates.append(("UP", up[0], up[1], up[2], i))
        if dn is not None:
            candidates.append(("DOWN", dn[0], dn[1], dn[2], i))
        if not candidates:
            continue
        candidates.sort(key=lambda x: (x[1], x[2]))
        entry_side, entry_t, entry_fill, _bid_entry, entry_idx_right = candidates[0]
        break

    if entry_side is None:
        return {"slug": slug, "traded": 0, "reason": "no_trade"}

    shares = float(stake_usdc) / float(entry_fill)
    t_min_stop = float(entry_t)

    for j in range(max(1, int(entry_idx_right)), len(seq)):
        s0 = seq[j - 1]
        s1 = seq[j]
        if float(s1.t) < t_min_stop - EPS:
            continue
        if float(s0.t) > close_epoch + 1:
            break
        if entry_side == "UP":
            stop_hit = segment_first_stop_w(s0.t, s1.t, s0.up_bid, s1.up_bid, s0.up_bid, s1.up_bid, t_min_stop, float(stop_price))
        else:
            stop_hit = segment_first_stop_w(s0.t, s1.t, s0.dn_bid, s1.dn_bid, s0.dn_bid, s1.dn_bid, t_min_stop, float(stop_price))
        if stop_hit is None:
            continue
        t_stop, _trigger_px_stop, fill_px_stop = stop_hit
        exit_price = max(float(fill_px_stop), float(stop_price) - float(stop_slip))
        pnl = shares * max(0.0, exit_price) - float(stake_usdc)
        return {
            "slug": slug, "traded": 1, "side": entry_side, "entry_epoch": float(entry_t), "fill": float(entry_fill),
            "shares": float(shares), "exit_epoch": float(t_stop), "exit_price": float(exit_price), "pnl_usdc": float(pnl),
            "reason": "stop", "stop_price": float(stop_price), "strike": strike,
            "entry_price_max": float(entry_price_max), "entry_floor": float(entry_floor), "last_window_sec": int(last_window_sec),
            "stop_slip": float(stop_slip),
        }

    official_winner = winner_map.get(slug)
    if official_winner not in ("UP", "DOWN"):
        return {"slug": slug, "traded": 0, "reason": "no_official_settle"}
    settle_price = 1.0 if entry_side == official_winner else 0.0
    pnl = shares * settle_price - float(stake_usdc)
    return {
        "slug": slug, "traded": 1, "side": entry_side, "entry_epoch": float(entry_t), "fill": float(entry_fill),
        "shares": float(shares), "exit_epoch": float(close_epoch), "exit_price": float(settle_price), "pnl_usdc": float(pnl),
        "reason": "settle_win" if settle_price > 0 else "settle_loss", "winner": official_winner,
        "strike": strike, "entry_price_max": float(entry_price_max), "entry_floor": float(entry_floor),
        "last_window_sec": int(last_window_sec), "stop_price": float(stop_price), "stop_slip": float(stop_slip),
    }


def evaluate_combo_parallel(records, seq_map, start_epoch_map, strike_map, ext, winner_map, interval_sec, stake_usdc,
                            entry_price_max, entry_floor, last_window_sec, stop_price, stop_slip, min_prob, max_prob,
                            spread_max, tie_wins, min_trades, include_trades=False):
    trades = []
    no_trade = 0
    skipped = Counter()
    for slug, seq in seq_map.items():
        start_epoch = start_epoch_map.get(slug)
        if start_epoch is None:
            continue
        result = simulate_one_market_settle(
            seq, start_epoch, strike_map, ext, slug=slug, interval_sec=interval_sec, stake_usdc=stake_usdc,
            winner_map=winner_map, entry_price_max=entry_price_max, entry_floor=entry_floor,
            last_window_sec=last_window_sec, stop_price=stop_price, stop_slip=stop_slip,
            min_prob=min_prob, max_prob=max_prob, spread_max=spread_max, tie_wins=tie_wins)
        if result.get("traded", 0) == 0:
            if result.get("reason") == "no_trade":
                no_trade += 1
            else:
                skipped[result.get("reason", "skip")] += 1
        else:
            trades.append(result)

    if len(trades) < int(min_trades):
        return None
    pnls = [float(t["pnl_usdc"]) for t in trades]
    avg_pnl = sum(pnls) / len(pnls)
    reasons = Counter(t["reason"] for t in trades)
    out = {
        "entry_price_max": float(entry_price_max), "entry_floor": float(entry_floor),
        "last_window_sec": int(last_window_sec), "stop_price": float(stop_price), "stop_slip": float(stop_slip),
        "n": len(trades), "avg_pnl_usdc": float(avg_pnl), "avg_return": float(avg_pnl) / float(stake_usdc) if stake_usdc > 0 else 0.0,
        "win_rate": sum(1 for x in pnls if x > 0) / len(pnls),
        "avg_hold_sec": sum(float(t["exit_epoch"]) - float(t["entry_epoch"]) for t in trades) / len(trades),
        "no_trade": int(no_trade), "no_trade_rate": float(no_trade) / len(records) if records else 0.0,
        "cnt_stop": int(reasons.get("stop", 0)), "cnt_settle_win": int(reasons.get("settle_win", 0)),
        "cnt_settle_loss": int(reasons.get("settle_loss", 0)), "skipped": dict(skipped),
    }
    if include_trades:
        out["trades"] = trades
    return out


_G_RECORDS = _G_SEQ_MAP = _G_START_EPOCH_MAP = _G_STRIKE_MAP = _G_EXT = _G_WINNER_MAP = None
_G_INTERVAL_SEC = _G_STAKE_USDC = _G_STOP_SLIP = _G_MIN_PROB = _G_MAX_PROB = _G_SPREAD_MAX = _G_TIE_WINS = _G_MIN_TRADES = None


def _warmup_numba() -> None:
    clamp(0.5, 0.0, 1.0)
    lerp(0.0, 1.0, 0.5)
    interval_for_linear_between(0.9, 0.95, 0.8, 0.97)
    interval_for_linear_leq(0.9, 0.95, 0.92)


def _init_worker(records, seq_map, start_epoch_map, strike_map, ext, winner_map, interval_sec, stake_usdc,
                 stop_slip, min_prob, max_prob, spread_max, tie_wins, min_trades) -> None:
    global _G_RECORDS, _G_SEQ_MAP, _G_START_EPOCH_MAP, _G_STRIKE_MAP, _G_EXT, _G_WINNER_MAP
    global _G_INTERVAL_SEC, _G_STAKE_USDC, _G_STOP_SLIP, _G_MIN_PROB, _G_MAX_PROB, _G_SPREAD_MAX, _G_TIE_WINS, _G_MIN_TRADES
    _G_RECORDS, _G_SEQ_MAP, _G_START_EPOCH_MAP, _G_STRIKE_MAP, _G_EXT, _G_WINNER_MAP = records, seq_map, start_epoch_map, strike_map, ext, winner_map
    _G_INTERVAL_SEC, _G_STAKE_USDC = interval_sec, stake_usdc
    _G_STOP_SLIP, _G_MIN_PROB, _G_MAX_PROB, _G_SPREAD_MAX, _G_TIE_WINS, _G_MIN_TRADES = stop_slip, min_prob, max_prob, spread_max, tie_wins, min_trades
    _warmup_numba()


def _worker_wrapper(combo: Tuple[float, float, int, float]):
    entry_floor, entry_price_max, last_window_sec, stop_price = combo
    if stop_price > entry_floor:
        return None
    if entry_floor > entry_price_max:
        return None
    return evaluate_combo_parallel(
        _G_RECORDS, _G_SEQ_MAP, _G_START_EPOCH_MAP, _G_STRIKE_MAP, _G_EXT, _G_WINNER_MAP,
        _G_INTERVAL_SEC, _G_STAKE_USDC, entry_price_max, entry_floor, last_window_sec, stop_price,
        _G_STOP_SLIP, _G_MIN_PROB, _G_MAX_PROB, _G_SPREAD_MAX, _G_TIE_WINS, _G_MIN_TRADES, include_trades=False)


def parse_maybe_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x.strip() or "[]")
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return []


def parse_ts_to_epoch(ts_str: str) -> int:
    return int(dt.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=UTC).timestamp())


def load_from_logs(log_paths):
    records, start_epoch_map, strike_map, ext = defaultdict(list), {}, {}, ExtSeries()
    for lp in log_paths:
        with open(lp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = EXTHIST_RE.search(line)
                if m:
                    ext.add(parse_ts_to_epoch(m.group("ts")), float(m.group("px")))
                    continue
                m = STRIKE_RE.search(line)
                if m:
                    slug = m.group("slug")
                    strike_map[slug] = float(m.group("strike"))
                    start_epoch_map[slug] = int(m.group("start_epoch"))
                    continue
                m = SNAP_RE.search(line)
                if not m:
                    continue
                slug = m.group("slug")
                start_epoch_map[slug] = int(m.group("start_epoch"))
                records[slug].append(Snap(parse_ts_to_epoch(m.group("ts")), float(m.group("up_bid")), float(m.group("up_ask")), float(m.group("dn_bid")), float(m.group("dn_ask"))))
    for slug in list(records.keys()):
        records[slug].sort(key=lambda s: s.t)
    return records, start_epoch_map, strike_map, ext


def window_seq(snaps: List[Snap], start_epoch: int, close_epoch: int) -> List[Snap]:
    return [s for s in snaps if start_epoch <= s.t <= close_epoch]


GAMMA_BASE_DEFAULT = "https://gamma-api.polymarket.com"


def gamma_get_market_by_slug(slug: str, *, gamma_base: str, timeout: float = 10.0, retries: int = 3, backoff: float = 0.6):
    base = (gamma_base or GAMMA_BASE_DEFAULT).rstrip("/")
    url = f"{base}/markets/slug/{urllib.parse.quote(slug, safe='')}"
    for k in range(max(1, int(retries))):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "paperbot-review/1.0", "Accept": "application/json"}, method="GET")
            with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            if getattr(e, "code", None) == 404:
                return None
        except Exception:
            pass
        time.sleep(backoff * (1.6 ** k) + 0.2)
    return None


def gamma_official_winner_updown(mkt: dict):
    if not isinstance(mkt, dict):
        return None
    outcomes = [str(x) for x in parse_maybe_list(mkt.get("outcomes"))]
    prices = parse_maybe_list(mkt.get("outcomePrices"))
    if len(outcomes) != len(prices):
        return None
    for i, p in enumerate(prices):
        try:
            if Decimal(str(p)) >= Decimal("0.9999"):
                w = outcomes[i].strip().upper()
                if w == "UP" or w.startswith("UP"):
                    return "UP"
                if w == "DOWN" or w.startswith("DOWN") or w.startswith("DO"):
                    return "DOWN"
        except Exception:
            continue
    return None


def build_official_winner_map(slugs: List[str], *, gamma_base: str, timeout: float = 10.0, retries: int = 3):
    out = {}
    for i, slug in enumerate(slugs, 1):
        mkt = gamma_get_market_by_slug(slug, gamma_base=gamma_base, timeout=timeout, retries=retries)
        w = gamma_official_winner_updown(mkt) if mkt else None
        if w in ("UP", "DOWN"):
            out[slug] = w
        if i % 50 == 0 or i == len(slugs):
            print(f"[Gamma] fetched {i}/{len(slugs)} winners ok={len(out)}", flush=True)
    return out


def load_winner_cache(path: Path):
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if v in ("UP", "DOWN")}
    except Exception:
        pass
    return {}


def save_winner_cache(path: Path, winner_map: Dict[str, str]) -> None:
    try:
        path.write_text(json.dumps(winner_map, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def frange(a: float, b: float, step: float) -> List[float]:
    out = []
    x = a
    while x <= b + 1e-12:
        out.append(round(x, 10))
        x += step
    return out


def parse_csv_floats(v: str) -> List[float]:
    if not v:
        return []
    return [float(x.strip()) for x in v.split(",") if x.strip()]


def parse_csv_ints(v: str) -> List[int]:
    if not v:
        return []
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", nargs="+", required=True)
    ap.add_argument("--interval-sec", type=int, default=900)
    ap.add_argument("--stake-usdc", type=float, default=5.0)
    ap.add_argument("--min-prob", type=float, default=0.93)
    ap.add_argument("--max-prob", type=float, default=0.99)
    ap.add_argument("--spread-max", type=float, default=0.05)
    ap.add_argument("--tie-wins", type=str, default="DOWN", choices=["UP", "DOWN"])

    # backward compatible entry floor args
    ap.add_argument("--entry-min", type=float, default=0.80, help="[兼容] 等同 --entry-floor-min")
    ap.add_argument("--entry-max", type=float, default=0.90, help="[兼容] 等同 --entry-floor-max")
    ap.add_argument("--entry-step", type=float, default=0.01, help="[兼容] 等同 --entry-floor-step")
    ap.add_argument("--entry-floor-min", type=float, default=None)
    ap.add_argument("--entry-floor-max", type=float, default=None)
    ap.add_argument("--entry-floor-step", type=float, default=None)

    # support entry_price_max sweep
    ap.add_argument("--fixed-entry-max", type=float, default=0.97)
    ap.add_argument("--entry-price-max-min", type=float, default=None)
    ap.add_argument("--entry-price-max-max", type=float, default=None)
    ap.add_argument("--entry-price-max-step", type=float, default=None)

    ap.add_argument("--window-min", type=int, default=30)
    ap.add_argument("--window-max", type=int, default=180)
    ap.add_argument("--window-step", type=int, default=30)
    ap.add_argument("--stop-min", type=float, default=0.80)
    ap.add_argument("--stop-max", type=float, default=0.95)
    ap.add_argument("--stop-step", type=float, default=0.01)
    ap.add_argument("--stop-slip", type=float, default=0.0)
    ap.add_argument("--gamma-base", type=str, default=GAMMA_BASE_DEFAULT)
    ap.add_argument("--gamma-timeout", type=float, default=10.0)
    ap.add_argument("--gamma-retries", type=int, default=3)
    ap.add_argument("--winner-cache", type=str, default="official_winners.json")
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--out", type=str, default="grid_results.csv")
    ap.add_argument("--out-trades", type=str, default="best_trades_detail.csv")
    ap.add_argument("--out-trades-combos", type=str, default="", help="指定组合逐笔明细输出 CSV")
    ap.add_argument("--detail-entry-floors", type=str, default="", help="逗号分隔，例如: 0.86,0.88")
    ap.add_argument("--detail-entry-maxes", type=str, default="", help="逗号分隔，例如: 0.96,0.97")
    ap.add_argument("--detail-windows", type=str, default="", help="逗号分隔，例如: 60,90,120")
    ap.add_argument("--detail-stops", type=str, default="", help="逗号分隔，例如: 0.84,0.85")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    records, start_epoch_map, strike_map, ext = load_from_logs(args.log)
    if not records:
        raise SystemExit("No snapshots found.")
    slugs = sorted(list(records.keys()))

    cache_path = Path(args.winner_cache)
    winner_map = load_winner_cache(cache_path)
    missing = [s for s in slugs if s not in winner_map]
    if missing:
        fetched = build_official_winner_map(missing, gamma_base=args.gamma_base, timeout=args.gamma_timeout, retries=args.gamma_retries)
        winner_map.update(fetched)
        save_winner_cache(cache_path, winner_map)

    seq_map = {}
    for slug, snaps in records.items():
        st = start_epoch_map.get(slug)
        if st is None:
            continue
        seq_map[slug] = window_seq(snaps, st, st + int(args.interval_sec))

    ef_min = args.entry_floor_min if args.entry_floor_min is not None else args.entry_min
    ef_max = args.entry_floor_max if args.entry_floor_max is not None else args.entry_max
    ef_step = args.entry_floor_step if args.entry_floor_step is not None else args.entry_step
    entry_floor_list = frange(ef_min, ef_max, ef_step)

    if args.entry_price_max_min is not None and args.entry_price_max_max is not None and args.entry_price_max_step is not None:
        entry_max_list = frange(args.entry_price_max_min, args.entry_price_max_max, args.entry_price_max_step)
    else:
        entry_max_list = [args.fixed_entry_max]

    window_list = list(range(args.window_min, args.window_max + 1, args.window_step))
    stop_list = frange(args.stop_min, args.stop_max, args.stop_step)

    combo_args = []
    for ef in entry_floor_list:
        for em in entry_max_list:
            for ws in window_list:
                for sp in stop_list:
                    if sp <= ef and ef <= em:
                        combo_args.append((ef, em, ws, sp))

    print(f"[Grid] Starting multiprocessing grid search with {len(combo_args)} combos", flush=True)
    t0 = time.time()
    all_metrics = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(records, seq_map, start_epoch_map, strike_map, ext, winner_map,
                                       args.interval_sec, args.stake_usdc, args.stop_slip, args.min_prob,
                                       args.max_prob, args.spread_max, args.tie_wins, args.min_trades)) as executor:
        for k, m in enumerate(executor.map(_worker_wrapper, combo_args, chunksize=20), 1):
            if m:
                all_metrics.append(m)
            if args.progress_every > 0 and k % args.progress_every == 0:
                elapsed = time.time() - t0
                print(f"[Grid] {k}/{len(combo_args)} done. Speed={k/max(elapsed, 1e-9):.1f} combo/s", flush=True)

    if not all_metrics:
        raise SystemExit("No combos met min_trades.")

    all_metrics.sort(key=lambda x: (x["avg_pnl_usdc"], x["n"]), reverse=True)
    best = all_metrics[0]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["entry_floor", "entry_price_max", "last_window_sec", "stop_price", "stop_slip", "n", "avg_pnl_usdc", "avg_return", "win_rate", "avg_hold_sec", "no_trade", "no_trade_rate", "cnt_stop", "cnt_settle_win", "cnt_settle_loss"])
        for m in all_metrics:
            w.writerow([f"{m['entry_floor']:.6f}", f"{m['entry_price_max']:.6f}", m["last_window_sec"], f"{m['stop_price']:.6f}", f"{m['stop_slip']:.6f}", m["n"], f"{m['avg_pnl_usdc']:.8f}", f"{m['avg_return']:.8f}", f"{m['win_rate']:.8f}", f"{m['avg_hold_sec']:.3f}", m["no_trade"], f"{m['no_trade_rate']:.8f}", m["cnt_stop"], m["cnt_settle_win"], m["cnt_settle_loss"]])

    best_detail = evaluate_combo_parallel(records, seq_map, start_epoch_map, strike_map, ext, winner_map,
                                          args.interval_sec, args.stake_usdc, best["entry_price_max"], best["entry_floor"],
                                          best["last_window_sec"], best["stop_price"], best["stop_slip"], args.min_prob,
                                          args.max_prob, args.spread_max, args.tie_wins, args.min_trades, include_trades=True)

    def dump_trade_rows(path: str, rows: List[dict]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["slug", "side", "reason", "entry_epoch", "fill", "shares", "exit_epoch", "exit_price", "pnl_usdc", "strike", "winner", "entry_floor", "entry_price_max", "last_window_sec", "stop_price", "stop_slip"])
            for t in rows:
                w.writerow([t.get("slug"), t.get("side"), t.get("reason"), f"{float(t.get('entry_epoch', 0.0)):.3f}", f"{float(t.get('fill', 0.0)):.6f}", f"{float(t.get('shares', 0.0)):.6f}", f"{float(t.get('exit_epoch', 0.0)):.3f}", f"{float(t.get('exit_price', 0.0)):.6f}", f"{float(t.get('pnl_usdc', 0.0)):.8f}", "" if t.get("strike") is None else f"{float(t.get('strike')):.4f}", t.get("winner", ""), f"{float(t.get('entry_floor', 0.0)):.6f}", f"{float(t.get('entry_price_max', 0.0)):.6f}", int(t.get("last_window_sec", 0)), f"{float(t.get('stop_price', 0.0)):.6f}", f"{float(t.get('stop_slip', 0.0)):.6f}"])

    if best_detail and "trades" in best_detail:
        dump_trade_rows(args.out_trades, best_detail["trades"])

    if args.out_trades_combos:
        filt_floors = set(parse_csv_floats(args.detail_entry_floors))
        filt_maxes = set(parse_csv_floats(args.detail_entry_maxes))
        filt_windows = set(parse_csv_ints(args.detail_windows))
        filt_stops = set(parse_csv_floats(args.detail_stops))

        def hit(v, filt, tol=1e-9):
            return (not filt) or any(abs(v - x) <= tol for x in filt)

        combo_trades = []
        for m in all_metrics:
            if not (hit(m["entry_floor"], filt_floors) and hit(m["entry_price_max"], filt_maxes) and
                    ((not filt_windows) or m["last_window_sec"] in filt_windows) and hit(m["stop_price"], filt_stops)):
                continue
            detail = evaluate_combo_parallel(records, seq_map, start_epoch_map, strike_map, ext, winner_map,
                                             args.interval_sec, args.stake_usdc, m["entry_price_max"], m["entry_floor"],
                                             m["last_window_sec"], m["stop_price"], m["stop_slip"], args.min_prob,
                                             args.max_prob, args.spread_max, args.tie_wins, args.min_trades, include_trades=True)
            if detail and "trades" in detail:
                combo_trades.extend(detail["trades"])
        dump_trade_rows(args.out_trades_combos, combo_trades)

    print(f"Saved summary: {args.out}")
    print(f"Saved best trades: {args.out_trades}")
    if args.out_trades_combos:
        print(f"Saved requested combo trades: {args.out_trades_combos}")


if __name__ == "__main__":
    main()
