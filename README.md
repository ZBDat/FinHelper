# FinHelper 全球指数监控（日K）

## 功能
- 监控四大市场：A股、美股、港股、欧洲。
- 市场拆解：
  - A股：上证、深证、创业板
  - 美股：标普500、纳斯达克、道琼斯
  - 港股：恒生、恒生科技（代理）
  - 欧洲：Euro Stoxx 50、DAX、CAC40
- 同时展示原始指数值与统一计量（CNY）标准化值。
- 统一单位对比曲线（Rebased=100，日K）。
- DeepSeek 资金流分析接口预留，支持 API Key 缺失降级。

## 启动
```bash
python /home/runner/work/FinHelper/FinHelper/portfolio_server.py --host 127.0.0.1 --port 8877
```
访问：`http://127.0.0.1:8877`

## 配置
- `DEEPSEEK_API_KEY`：DeepSeek API Key（可选，未配置时返回降级结果）
- `DEEPSEEK_API_URL`：DeepSeek 接口地址（默认 `https://api.deepseek.com/chat/completions`）

## 统一计量说明
- 基准币种：CNY
- 标准化公式：`normalized_close = raw_close * fx_to_cny / stat_divisor`
- 时间粒度：日K
