from __future__ import annotations

import argparse
import json
import logging
import queue
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
DEFAULT_HEADERS = {"User-Agent": "FinHelper/1.0"}
MIN_POLL_INTERVAL = 5
DEFAULT_GOLD_TEST_COST = 2300.0


SYMBOL_ALIASES = {
    "现货黄金": "XAUUSD=X",
    "黄金": "XAUUSD=X",
    "spot gold": "XAUUSD=X",
    "xauusd": "XAUUSD=X",
    "xauusd=x": "XAUUSD=X",
    "gold": "XAUUSD=X",
}


@dataclass
class Holding:
    symbol: str
    display_name: str
    quantity: float
    cost_price: float
    imported_at: float


class PortfolioState:
    def __init__(self, poll_interval: int = 10) -> None:
        self.poll_interval = max(MIN_POLL_INTERVAL, poll_interval)
        self.holdings: Dict[str, Holding] = {}
        self.price_history: Dict[str, List[Dict[str, float]]] = {}
        self.day_open: Dict[str, float] = {}
        self.latest_price: Dict[str, float] = {}
        self.interaction_stub = {
            "has_future_channel": True,
            "message": "预留客户交互通道：后续可接入登录态、会话消息、指令下发，不影响图表渲染。",
            "capabilities": ["notifications", "commands", "chat"],
        }
        self._subs: List[queue.Queue[str]] = []
        self._lock = threading.Lock()

    @staticmethod
    def normalize_symbol(value: str) -> str:
        key = (value or "").strip()
        lower = key.lower()
        if lower in SYMBOL_ALIASES:
            return SYMBOL_ALIASES[lower]
        return key

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
                holding = Holding(
                    symbol=symbol,
                    display_name=source,
                    quantity=quantity,
                    cost_price=cost_price,
                    imported_at=time.time(),
                )
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
        payload = f"event: snapshot\ndata: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
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

    def fetch_curve(self, symbol: str) -> Dict[str, Any]:
        encoded = urllib.parse.quote(symbol)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?interval=1m&range=1d"
        req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = resp.read()
        except urllib.error.URLError as exc:
            raise ValueError(f"{symbol} 行情获取失败: 网络连接异常") from exc
        parsed = json.loads(body.decode("utf-8"))
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
            if idx >= len(closes):
                continue
            px = closes[idx]
            if px is None:
                continue
            points.append({"ts": int(ts), "price": float(px)})

        meta = result.get("meta") or {}
        current = float(meta.get("regularMarketPrice") or (points[-1]["price"] if points else 0.0))
        day_open = points[0]["price"] if points else current
        if current <= 0:
            raise ValueError("No valid market price")
        return {"current": current, "day_open": day_open, "points": points}

    def update_market_data(self) -> None:
        with self._lock:
            symbols = list(self.holdings.keys())
        if not symbols:
            return

        for symbol in symbols:
            try:
                curve = self.fetch_curve(symbol)
            except Exception:
                logging.exception("Failed to fetch curve for symbol %s", symbol)
                continue

            now_ts = int(time.time())
            latest_point = {"ts": now_ts, "price": float(curve["current"])}
            with self._lock:
                self.latest_price[symbol] = float(curve["current"])
                self.day_open[symbol] = float(curve["day_open"])
                history = self.price_history.setdefault(symbol, [])
                for point in curve["points"]:
                    if history and point["ts"] <= history[-1]["ts"]:
                        continue
                    history.append(point)
                if not history or latest_point["ts"] > history[-1]["ts"]:
                    history.append(latest_point)
                elif history:
                    history[-1] = latest_point
                self.price_history[symbol] = history[-1500:]

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

        if self.path == "/api/events":
            self._serve_events()
            return

        if self.path == "/api/client-interaction":
            self._send_json(200, {"ok": True, "data": STATE.interaction_stub})
            return

        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/holdings/import":
            self._send_json(404, {"ok": False, "error": "Not found"})
            return

        try:
            payload = self._read_json()
            rows = payload.get("rows") or []
            replace = bool(payload.get("replace", False))
            result = STATE.import_rows(rows=rows, replace=replace)
            STATE.update_market_data()
            STATE.broadcast_snapshot()
            self._send_json(200, {"ok": True, "result": result, "data": STATE.snapshot()})
        except ValueError as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            logging.exception("Import holdings failed")
            self._send_json(400, {"ok": False, "error": f"导入失败: {type(exc).__name__}"})

    def _serve_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        sub = STATE.register_subscriber()
        try:
            initial = f"event: snapshot\ndata: {json.dumps(STATE.snapshot(), ensure_ascii=False)}\n\n"
            self.wfile.write(initial.encode("utf-8"))
            self.wfile.flush()
            while True:
                try:
                    msg = sub.get(timeout=20)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
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
            STATE.broadcast_snapshot()
        except Exception:
            logging.exception("Market loop iteration failed")
        time.sleep(STATE.poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime portfolio tracker backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--poll-seconds", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    STATE.poll_interval = max(MIN_POLL_INTERVAL, args.poll_seconds)

    preload = [{"symbol_or_name": "现货黄金", "quantity": 1, "cost_price": DEFAULT_GOLD_TEST_COST}]
    STATE.import_rows(preload, replace=True)
    STATE.update_market_data()

    t = threading.Thread(target=market_loop, daemon=True)
    t.start()

    server = ThreadingHTTPServer((args.host, args.port), PortfolioHandler)
    print(f"Portfolio service ready at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
