from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from analysis.ui_mlp_service import TrainedBundle, predict_with_bundle, train_from_uploaded_data


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data" / "ui_mlp_model.pth"
_TRAINED_BUNDLE: Optional[TrainedBundle] = None
_MODEL_LOCK = threading.Lock()


class TrainApiHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/api/status":
            self._send_json(404, {"ok": False, "error": "Not found"})
            return

        global _TRAINED_BUNDLE
        with _MODEL_LOCK:
            bundle = _TRAINED_BUNDLE
        if bundle is None:
            self._send_json(200, {"ok": True, "has_model": False})
            return

        self._send_json(
            200,
            {
                "ok": True,
                "has_model": True,
                "model_path": bundle.model_path,
                "setup": bundle.report.get("setup", {}),
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        global _TRAINED_BUNDLE
        try:
            payload = self._read_json()
            if self.path == "/api/train":
                strategy = payload.get("strategy", "mlp")
                if strategy != "mlp":
                    self._send_json(400, {"ok": False, "error": "Only MLP strategy is currently supported"})
                    return
                datasets = payload.get("datasets", [])
                params = payload.get("params", {})
                bundle = train_from_uploaded_data(datasets=datasets, params=params, save_path=MODEL_PATH)
                with _MODEL_LOCK:
                    _TRAINED_BUNDLE = bundle
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "report": bundle.report,
                        "model_path": bundle.model_path,
                    },
                )
                return

            if self.path == "/api/predict":
                with _MODEL_LOCK:
                    bundle = _TRAINED_BUNDLE
                if bundle is None:
                    self._send_json(400, {"ok": False, "error": "Please train a model first"})
                    return
                datasets = payload.get("datasets", [])
                prediction = predict_with_bundle(bundle, datasets)
                self._send_json(200, {"ok": True, "prediction": prediction})
                return

            self._send_json(404, {"ok": False, "error": "Not found"})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training API server for index.html")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), TrainApiHandler)
    print(f"Training API ready at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
