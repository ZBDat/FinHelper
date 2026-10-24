from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent
STATIC_INDEX = ROOT / "index.html"
DEFAULT_HEADERS = {"User-Agent": "FinHelper/2.0"}
MIN_POLL_INTERVAL = 5
DEFAULT_GOLD_TEST_COST = 2300.0
MAX_ERROR_MESSAGES = 4
MARKET_RANGE = "6mo"
MARKET_INTERVAL = "1d"

YAHOO_CHART_ENDPOINTS = (
    f"https://query1.finance.yahoo.com/v8/finance/chart/{{encoded}}?interval={MARKET_INTERVAL}&range={MARKET_RANGE}",
    f"https://query2.finance.yahoo.com/v8/finance/chart/{{encoded}}?interval={MARKET_INTERVAL}&range={MARKET_RANGE}",
)
YAHOO_QUOTE_ENDPOINT = "https://query2.finance.yahoo.com/v7/finance/quote?symbols={encoded}"
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_TIMEOUT_SECONDS = 8
DEEPSEEK_MIN_INTERVAL_SECONDS = 2.0

PREFERRED_PRICE_MIN = 100.0
PREFERRED_PRICE_MAX = 10000.0
FALLBACK_PRICE_MAX = 1000000.0

SYMBOL_ALIASES = {
    "现货黄金": "XAUUSD=X",
    "黄金": "XAUUSD=X",
    "spot gold": "XAUUSD=X",
    "xauusd": "XAUUSD=X",
    "xauusd=x": "XAUUSD=X",
    "gold": "XAUUSD=X",
}

FETCH_EXCEPTIONS = (
    ValueError,
    urllib.error.URLError,
    urllib.error.HTTPError,
    json.JSONDecodeError,
    KeyError,
    TypeError,
)


def _tail_errors(errors: List[str]) -> List[str]:
    return errors[-MAX_ERROR_MESSAGES:] if len(errors) > MAX_ERROR_MESSAGES else errors


@dataclass
class Holding:
    symbol: str
    display_name: str
    quantity: float
    cost_price: float
    imported_at: float


@dataclass(frozen=True)
class IndexConfig:
    code: str
    name: str
    market: str
    currency: str
    stat_mode: str
    stat_divisor: float = 1.0


MARKET_CONFIGS: Dict[str, Dict[str, str]] = {
    "cn": {"name": "A股", "description": "中国大陆主要指数（日K）"},
    "us": {"name": "美股", "description": "美国主要指数（日K）"},
    "hk": {"name": "港股", "description": "香港主要指数（日K）"},
    "eu": {"name": "欧洲", "description": "欧洲主要指数（日K）"},
}

INDEX_CONFIGS: List[IndexConfig] = [
    IndexConfig("000001.SS", "上证指数", "cn", "CNY", "index_points", 1.0),
    IndexConfig("399001.SZ", "深证成指", "cn", "CNY", "index_points", 1.0),
    IndexConfig("399006.SZ", "创业板指", "cn", "CNY", "index_points", 1.0),
    IndexConfig("^GSPC", "标普500", "us", "USD", "index_points", 1.0),
    IndexConfig("^IXIC", "纳斯达克综合", "us", "USD", "index_points", 1.0),
    IndexConfig("^DJI", "道琼斯工业", "us", "USD", "index_points", 1.0),
    IndexConfig("^HSI", "恒生指数", "hk", "HKD", "index_points", 1.0),
    IndexConfig("^HSCI", "恒生科技(代理)", "hk", "HKD", "index_points", 1.0),
    IndexConfig("^STOXX50E", "Euro Stoxx 50", "eu", "EUR", "index_points", 1.0),
    IndexConfig("^GDAXI", "DAX", "eu", "EUR", "index_points", 1.0),
    IndexConfig("^FCHI", "CAC40", "eu", "EUR", "index_points", 1.0),
]

INDEX_BY_CODE = {item.code: item for item in INDEX_CONFIGS}


class DeepSeekFlowAnalyzer:
    def __init__(self) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self._lock = threading.Lock()
        self._last_call_ts = 0.0

    def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            if now - self._last_call_ts < DEEPSEEK_MIN_INTERVAL_SECONDS:
                return {
                    "ok": False,
                    "degraded": True,
                    "error": "请求过于频繁，请稍后重试",
                    "last_updated": int(now),
                }
            self._last_call_ts = now

        if not self.api_key:
            return {
                "ok": False,
                "degraded": True,
                "error": "DEEPSEEK_API_KEY 未配置，已返回降级结果",
                "analysis": "未配置 DeepSeek API Key，暂无法生成资金流解读。",
                "last_updated": int(now),
            }

        market = str(payload.get("market") or "all")
        summary = str(payload.get("summary") or "")[:500]
        prompt = (
            "请基于以下指数快照做资金流向分析，输出简洁中文结论，包含风险提示。\\n"
            f"market={market}\\n"
            f"summary={summary}"
        )
        req_payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是金融市场分析助手，输出客观简洁结论。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        }

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "FinHelper/2.0",
        }

        logging.info("DeepSeek request: market=%s summary_len=%s", market, len(summary))
        req = urllib.request.Request(
            DEEPSEEK_API_URL,
            data=json.dumps(req_payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=DEEPSEEK_TIMEOUT_SECONDS) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            content = (((parsed.get("choices") or [{}])[0]).get("message") or {}).get("content")
            if not content:
                raise ValueError("empty deepseek response")
            return {
                "ok": True,
                "degraded": False,
                "analysis": content.strip(),
                "last_updated": int(time.time()),
            }
        except Exception as exc:  # pragma: no cover - network failure path
            logging.exception("DeepSeek request failed")
            return {
                "ok": False,
                "degraded": True,
                "error": f"DeepSeek 调用失败: {type(exc).__name__}",
                "analysis": "资金流分析暂不可用，请稍后重试。",
                "last_updated": int(time.time()),
            }


class PortfolioState:
    def __init__(self, poll_interval: int = 15) -> None:
        self.poll_interval = max(MIN_POLL_INTERVAL, poll_interval)
        self.holdings: Dict[str, Holding] = {}
        self.price_history: Dict[str, List[Dict[str, float]]] = {}
        self.day_open: Dict[str, float] = {}
        self.latest_price: Dict[str, float] = {}

        self.index_snapshots: Dict[str, Dict[str, Any]] = {}
        self.fx_rates: Dict[str, float] = {"CNY": 1.0, "USD": 7.2, "HKD": 0.92, "EUR": 7.8}
        self._market_last_update_ts = 0

        self.interaction_stub = {
            "has_future_channel": True,
            "message": "预留客户交互通道：后续可接入登录态、会话消息、指令下发，不影响图表渲染。",
            "capabilities": ["notifications", "commands", "chat"],
        }
        self._subs: List[queue.Queue[str]] = []
        self._lock = threading.Lock()
        self.deepseek = DeepSeekFlowAnalyzer()

    @staticmethod
    def normalize_symbol(value: str) -> str:
        key = (value or "").strip()
        lower = key.lower()
        return SYMBOL_ALIASES.get(lower, key)

    def import_rows(self, rows: List[Dict[str, Any]], replace: bool = False) -> Dict[str, Any]:
        imported = []
        with self._lock:
            if replace:
                self.holdings.clear()
            for row in rows:
                source = str(row.get("symbol_or_name", "")).strip()
                if not source:
                    continue
                symbol = self.normalize_symbol(source)
                quantity = float(row.get("quantity", 0) or 0)
                cost_price = float(row.get("cost_price", 0) or 0)
                if quantity <= 0:
                    continue
                holding = Holding(symbol=symbol, display_name=source, quantity=quantity, cost_price=cost_price, imported_at=time.time())
                self.holdings[symbol] = holding
                self.price_history.setdefault(symbol, [])
                imported.append({"symbol": symbol, "display_name": source, "quantity": quantity, "cost_price": cost_price})
        return {"imported": imported, "count": len(imported)}

    def register_subscriber(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=16)
        with self._lock:
            self._subs.append(q)
        return q

    def unregister_subscriber(self, q: queue.Queue[str]) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def broadcast_snapshot(self) -> None:
        snapshot = self.snapshot()
        payload = f"event: snapshot\\ndata: {json.dumps(snapshot, ensure_ascii=False)}\\n\\n"
        stale: List[queue.Queue[str]] = []
        with self._lock:
            targets = list(self._subs)
        for sub in targets:
            try:
                sub.put_nowait(payload)
            except queue.Full:
                stale.append(sub)
        for sub in stale:
            self.unregister_subscriber(sub)

    def _request_json(self, url: str) -> Dict[str, Any]:
        req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = resp.read()
        except urllib.error.URLError as exc:
            raise ValueError("网络连接异常") from exc
        return json.loads(body.decode("utf-8"))

    @staticmethod
    def _extract_reliable_prices(raw: str) -> Dict[str, Any]:
        values = []
        for token in re.split(r"[,\s~\"]+", raw):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
        preferred = [v for v in values if PREFERRED_PRICE_MIN <= v <= PREFERRED_PRICE_MAX]
        candidates = preferred or [v for v in values if 0 < v < FALLBACK_PRICE_MAX]
        if not candidates:
            raise ValueError("No numeric price in payload")
        current = candidates[0]
        day_open = candidates[1] if len(candidates) > 1 else current
        return {"current": current, "day_open": day_open, "points": []}

    def _fetch_quote_points(self, symbol: str) -> Dict[str, Any]:
        encoded = urllib.parse.quote(symbol)
        errors: List[str] = []
        for pattern in YAHOO_CHART_ENDPOINTS:
            try:
                parsed = self._request_json(pattern.format(encoded=encoded))
                chart = parsed.get("chart", {})
                if chart.get("error"):
                    raise ValueError(str(chart["error"]))
                result = (chart.get("result") or [None])[0]
                if not result:
                    raise ValueError("No chart result")
                timestamps = result.get("timestamp") or []
                quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
                closes = quote.get("close") or []
                points = []
                for idx, ts in enumerate(timestamps):
                    if idx >= len(closes) or closes[idx] is None:
                        continue
                    points.append({"ts": int(ts), "price": float(closes[idx])})
                meta = result.get("meta") or {}
                current = float(meta.get("regularMarketPrice") or (points[-1]["price"] if points else 0.0))
                if current <= 0 and points:
                    current = points[-1]["price"]
                if current <= 0:
                    raise ValueError("No valid market price")
                day_open = points[-1]["price"] if points else current
                previous = points[-2]["price"] if len(points) > 1 else day_open
                return {"current": current, "day_open": day_open, "points": points, "previous_close": previous}
            except FETCH_EXCEPTIONS as exc:
                errors.append(str(exc))

        parsed = self._request_json(YAHOO_QUOTE_ENDPOINT.format(encoded=encoded))
        result = ((parsed.get("quoteResponse") or {}).get("result") or [None])[0]
        if result:
            current = float(result.get("regularMarketPrice") or 0.0)
            day_open = float(result.get("regularMarketOpen") or current)
            prev = float(result.get("regularMarketPreviousClose") or day_open)
            if current > 0:
                return {"current": current, "day_open": day_open, "points": [], "previous_close": prev}

        detail = " | ".join(filter(None, _tail_errors(errors)))
        raise ValueError(f"{symbol} 行情获取失败: {detail or '网络连接异常'}")

    def _fetch_fx_rates(self) -> Dict[str, float]:
        rates = {"CNY": 1.0}
        mapping = {
            "USD": "CNY=X",
            "HKD": "HKDCNY=X",
            "EUR": "EURCNY=X",
        }
        for currency, symbol in mapping.items():
            try:
                quote = self._fetch_quote_points(symbol)
                rates[currency] = float(quote["current"])
            except Exception:
                logging.exception("Failed to fetch FX for %s", currency)
                rates[currency] = self.fx_rates.get(currency, 1.0)
        return rates

    @staticmethod
    def normalized_close(close: float, fx_to_cny: float, stat_divisor: float) -> float:
        divisor = stat_divisor if stat_divisor > 0 else 1.0
        return close * fx_to_cny / divisor

    def update_market_data(self) -> None:
        with self._lock:
            symbols = list(self.holdings.keys())
        if not symbols:
            return

        for symbol in symbols:
            try:
                curve = self._fetch_quote_points(symbol)
            except Exception:
                logging.exception("Failed to fetch curve for symbol %s", symbol)
                continue

            now_ts = int(time.time())
            latest_point = {"ts": now_ts, "price": float(curve["current"])}
            with self._lock:
                self.latest_price[symbol] = float(curve["current"])
                self.day_open[symbol] = float(curve["day_open"])
                history = self.price_history.setdefault(symbol, [])
                for point in curve.get("points", []):
                    if history and point["ts"] <= history[-1]["ts"]:
                        continue
                    history.append(point)
                if not history or latest_point["ts"] > history[-1]["ts"]:
                    history.append(latest_point)
                elif history:
                    history[-1] = latest_point
                self.price_history[symbol] = history[-1500:]

    def update_indices_data(self) -> None:
        rates = self._fetch_fx_rates()
        snapshots: Dict[str, Dict[str, Any]] = {}
        for cfg in INDEX_CONFIGS:
            try:
                data = self._fetch_quote_points(cfg.code)
            except Exception:
                logging.exception("Failed to fetch index %s", cfg.code)
                continue

            fx = rates.get(cfg.currency, 1.0)
            current = float(data["current"])
            previous = float(data.get("previous_close") or current)
            pct = (current - previous) / previous if previous else 0.0

            normalized_current = self.normalized_close(current, fx, cfg.stat_divisor)
            points = []
            first_normalized = None
            for point in data.get("points", []):
                normalized = self.normalized_close(float(point["price"]), fx, cfg.stat_divisor)
                if first_normalized is None and normalized > 0:
                    first_normalized = normalized
                rebased = (normalized / first_normalized * 100.0) if first_normalized else 100.0
                points.append({
                    "ts": int(point["ts"]),
                    "close": float(point["price"]),
                    "normalized_close": normalized,
                    "rebased_100": rebased,
                })

            snapshots[cfg.code] = {
                "code": cfg.code,
                "name": cfg.name,
                "market": cfg.market,
                "currency": cfg.currency,
                "stat_mode": cfg.stat_mode,
                "stat_divisor": cfg.stat_divisor,
                "fx_to_cny": fx,
                "raw_close": current,
                "normalized_close": normalized_current,
                "day_change_pct": pct,
                "points": points[-120:],
                "updated_at": int(time.time()),
            }

        with self._lock:
            self.fx_rates = rates
            self.index_snapshots = snapshots
            self._market_last_update_ts = int(time.time())

    def market_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snapshots = list(self.index_snapshots.values())
            rates = dict(self.fx_rates)
            last_ts = self._market_last_update_ts

        by_market: Dict[str, List[Dict[str, Any]]] = {key: [] for key in MARKET_CONFIGS.keys()}
        for item in snapshots:
            by_market.setdefault(item["market"], []).append(item)

        overview = []
        for key, conf in MARKET_CONFIGS.items():
            rows = by_market.get(key, [])
            if not rows:
                overview.append(
                    {
                        "market": key,
                        "name": conf["name"],
                        "description": conf["description"],
                        "index_count": 0,
                        "avg_raw_close": 0.0,
                        "avg_normalized_close": 0.0,
                        "avg_day_change_pct": 0.0,
                    }
                )
                continue
            avg_raw = sum(r["raw_close"] for r in rows) / len(rows)
            avg_normalized = sum(r["normalized_close"] for r in rows) / len(rows)
            avg_day_change = sum(r["day_change_pct"] for r in rows) / len(rows)
            overview.append(
                {
                    "market": key,
                    "name": conf["name"],
                    "description": conf["description"],
                    "index_count": len(rows),
                    "avg_raw_close": avg_raw,
                    "avg_normalized_close": avg_normalized,
                    "avg_day_change_pct": avg_day_change,
                }
            )

        comparison_curve = []
        market_curve_map: Dict[int, Dict[str, float]] = {}
        for key in MARKET_CONFIGS.keys():
            rows = by_market.get(key, [])
            for row in rows:
                for p in row.get("points", []):
                    ts = int(p["ts"])
                    market_curve_map.setdefault(ts, {}).setdefault(key, 0.0)
                    market_curve_map[ts][key] += float(p["rebased_100"])

        for ts in sorted(market_curve_map.keys()):
            record = {"ts": ts}
            for key in MARKET_CONFIGS.keys():
                rows = by_market.get(key, [])
                divisor = len(rows) if rows else 1
                record[key] = market_curve_map[ts].get(key, 0.0) / divisor
            comparison_curve.append(record)

        return {
            "updated_at": last_ts,
            "base_currency": "CNY",
            "fx_rates": rates,
            "overview": overview,
            "drilldown": by_market,
            "comparison_curve": comparison_curve[-120:],
            "normalization": {
                "formula": "normalized_close = raw_close * fx_to_cny / stat_divisor",
                "timeframe": "日K",
            },
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            holdings = list(self.holdings.values())
            latest = dict(self.latest_price)
            opens = dict(self.day_open)
            history_map = {k: list(v) for k, v in self.price_history.items()}

        assets = []
        total_current = 0.0
        total_cost = 0.0
        total_today_base = 0.0
        total_curve_map: Dict[int, float] = {}

        for holding in holdings:
            price = latest.get(holding.symbol, 0.0)
            open_price = opens.get(holding.symbol, price)
            current_value = price * holding.quantity
            cost_value = holding.cost_price * holding.quantity
            today_base = open_price * holding.quantity

            total_current += current_value
            total_cost += cost_value
            total_today_base += today_base

            raw_curve = history_map.get(holding.symbol, [])
            value_curve = []
            for point in raw_curve:
                point_value = point["price"] * holding.quantity
                ts = int(point["ts"])
                value_curve.append({"ts": ts, "value": point_value, "price": point["price"]})
                total_curve_map[ts] = total_curve_map.get(ts, 0.0) + point_value

            since_profit = current_value - cost_value
            since_return = (since_profit / cost_value) if cost_value > 0 else 0.0
            today_profit = current_value - today_base

            assets.append(
                {
                    "symbol": holding.symbol,
                    "display_name": holding.display_name,
                    "quantity": holding.quantity,
                    "cost_price": holding.cost_price,
                    "current_price": price,
                    "current_value": current_value,
                    "since_profit": since_profit,
                    "since_return_rate": since_return,
                    "today_profit": today_profit,
                    "curve": value_curve,
                }
            )

        portfolio_profit = total_current - total_cost
        portfolio_return = (portfolio_profit / total_cost) if total_cost > 0 else 0.0
        portfolio_today_profit = total_current - total_today_base
        total_curve = [{"ts": ts, "value": value} for ts, value in sorted(total_curve_map.items())]

        return {
            "ts": int(time.time()),
            "poll_interval": self.poll_interval,
            "assets": assets,
            "portfolio": {
                "total_current_value": total_current,
                "total_cost": total_cost,
                "since_profit": portfolio_profit,
                "since_return_rate": portfolio_return,
                "today_profit": portfolio_today_profit,
                "curve": total_curve,
            },
            "market_monitor": self.market_snapshot(),
            "interaction_stub": self.interaction_stub,
        }


STATE = PortfolioState()


class PortfolioHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            if not STATIC_INDEX.exists():
                self._send_json(404, {"ok": False, "error": "index.html not found"})
                return
            body = STATIC_INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/status":
            self._send_json(200, {"ok": True, "service": "portfolio", "poll_interval": STATE.poll_interval})
            return

        if self.path == "/api/snapshot":
            self._send_json(200, {"ok": True, "data": STATE.snapshot()})
            return

        if self.path == "/api/markets/overview":
            self._send_json(200, {"ok": True, "data": STATE.market_snapshot()})
            return

        if self.path.startswith("/api/markets/drilldown"):
            parsed = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(parsed.query)
            market = (q.get("market") or [""])[0]
            data = STATE.market_snapshot()
            rows = data["drilldown"].get(market, []) if market else data["drilldown"]
            self._send_json(200, {"ok": True, "market": market or None, "data": rows})
            return

        if self.path == "/api/events":
            self._serve_events()
            return

        if self.path == "/api/client-interaction":
            self._send_json(200, {"ok": True, "data": STATE.interaction_stub})
            return

        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/holdings/import":
            try:
                payload = self._read_json()
                rows = payload.get("rows") or []
                replace = bool(payload.get("replace", False))
                result = STATE.import_rows(rows=rows, replace=replace)
                STATE.update_market_data()
                STATE.broadcast_snapshot()
                self._send_json(200, {"ok": True, "result": result, "data": STATE.snapshot()})
                return
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:
                logging.exception("Import holdings failed")
                self._send_json(400, {"ok": False, "error": f"导入失败: {type(exc).__name__}"})
                return

        if self.path == "/api/deepseek/fund-flow":
            payload = self._read_json()
            result = STATE.deepseek.analyze(payload)
            self._send_json(200, {"ok": True, "data": result})
            return

        self._send_json(404, {"ok": False, "error": "Not found"})

    def _serve_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        sub = STATE.register_subscriber()
        try:
            initial = f"event: snapshot\\ndata: {json.dumps(STATE.snapshot(), ensure_ascii=False)}\\n\\n"
            self.wfile.write(initial.encode("utf-8"))
            self.wfile.flush()
            while True:
                try:
                    msg = sub.get(timeout=20)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\\n\\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(msg.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            STATE.unregister_subscriber(sub)


def market_loop() -> None:
    while True:
        try:
            STATE.update_market_data()
            STATE.update_indices_data()
            STATE.broadcast_snapshot()
        except Exception:
            logging.exception("Market loop iteration failed")
        time.sleep(STATE.poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime portfolio tracker backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--poll-seconds", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    STATE.poll_interval = max(MIN_POLL_INTERVAL, args.poll_seconds)

    preload = [{"symbol_or_name": "现货黄金", "quantity": 1, "cost_price": DEFAULT_GOLD_TEST_COST}]
    STATE.import_rows(preload, replace=True)
    STATE.update_market_data()
    STATE.update_indices_data()

    t = threading.Thread(target=market_loop, daemon=True)
    t.start()

    server = ThreadingHTTPServer((args.host, args.port), PortfolioHandler)
    print(f"Portfolio service ready at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
