import os

from portfolio_server import DeepSeekFlowAnalyzer, PortfolioState


def test_normalized_close_uses_fx_and_stat_divisor() -> None:
    value = PortfolioState.normalized_close(close=5000, fx_to_cny=7.2, stat_divisor=2)
    assert value == 18000


def test_market_snapshot_aggregates_by_market() -> None:
    state = PortfolioState()
    state.fx_rates = {"CNY": 1.0, "USD": 7.2, "HKD": 0.92, "EUR": 7.8}
    state.index_snapshots = {
        "000001.SS": {
            "market": "cn",
            "raw_close": 3000.0,
            "normalized_close": 3000.0,
            "day_change_pct": 0.01,
            "points": [{"ts": 1, "rebased_100": 100.0}],
        },
        "399001.SZ": {
            "market": "cn",
            "raw_close": 9000.0,
            "normalized_close": 9000.0,
            "day_change_pct": -0.01,
            "points": [{"ts": 1, "rebased_100": 102.0}],
        },
        "^GSPC": {
            "market": "us",
            "raw_close": 5000.0,
            "normalized_close": 36000.0,
            "day_change_pct": 0.02,
            "points": [{"ts": 1, "rebased_100": 101.0}],
        },
    }

    data = state.market_snapshot()
    overview = {row["market"]: row for row in data["overview"]}
    assert overview["cn"]["index_count"] == 2
    assert overview["cn"]["avg_raw_close"] == 6000.0
    assert overview["cn"]["avg_day_change_pct"] == 0.0
    assert overview["us"]["avg_normalized_close"] == 36000.0


def test_deepseek_fallback_when_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    analyzer = DeepSeekFlowAnalyzer()
    result = analyzer.analyze({"market": "us", "summary": "test"})
    assert result["degraded"] is True
    assert "API Key" in result["error"]
