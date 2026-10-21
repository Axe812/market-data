#!/usr/bin/env python3
"""market-data：全球质量价值筛选器（每个交易日在 GitHub Actions 上运行）

股票池只用公开标准：标普500、纳斯达克100、沪深300、港股主要公司、市值前列的加密货币。
1. 抓 1 年日线、5 年周线、汇率、指数
2. 基本面（雅虎财经）按 7 天轮换刷新，三个市场轮流排队
3. 算指标：涨跌、回撤、5 年价格分位、自由现金流收益率等
4. 筛选：质量门槛 ×（便宜 / 错杀）
5. 输出 data/latest/*，并按日期归档 data/history/

所有阈值集中在 RULES。数据来自免费接口，港股/A股基本面有缺口，仅供研究。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LATEST = DATA / "latest"
HISTORY = DATA / "history"
CFG = json.loads((ROOT / "config" / "universe.json").read_text())

RULES = {
    "roe_min": 0.15,             # 质量：净资产收益率 ≥ 15%
    "de_max": 100.0,             # 质量：负债/股东权益 ≤ 100%（雅虎以百分数给出）
    "op_margin_min": 0.10,       # 质量：营业利润率 ≥ 10%
    "fcf_yield_min": 0.05,       # 便宜：自由现金流收益率 ≥ 5%
    "fcf_yield_suspect": 0.25,   # 非金融股 FCF 收益率 > 25% 视为数据存疑
    "fwd_pe_max": 15.0,          # 便宜：预期市盈率 ≤ 15
    "pct5y_max": 0.20,           # 便宜：价格处于自身 5 年区间最低 20%
    "pb_max_fin": 1.2,           # 便宜（金融/地产）：PB ≤ 1.2
    "ret20_max": -0.15,          # 错杀：20 个交易日跌 ≥ 15%
    "dd52_max": -0.35,           # 错杀：距 52 周高点回撤 ≥ 35%
    "fund_ttl_days": 7,
}

NO_FCF_SECTORS = {"Financial Services", "Financials", "Real Estate"}
CYCLICAL_SECTORS = {"Basic Materials", "Energy"}
CYCLICAL_WORDS = ("Insurance", "Steel", "Aluminum", "Copper", "Gold", "Coal", "Oil", "Auto",
                  "Semiconductor", "Memory")

INFO_FIELDS = [
    "shortName", "currency", "financialCurrency", "marketCap",
    "trailingPE", "forwardPE", "priceToBook", "returnOnEquity", "debtToEquity",
    "freeCashflow", "operatingMargins", "profitMargins", "revenueGrowth",
    "earningsGrowth", "trailingAnnualDividendYield", "payoutRatio",
    "trailingEps", "forwardEps", "sector", "industry",
    "numberOfAnalystOpinions", "targetMeanPrice",
    "sharesOutstanding", "impliedSharesOutstanding", "netIncomeToCommon",
]


def log(*a):
    print(f"[{dt.datetime.utcnow():%H:%M:%S}]", *a, flush=True)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (pd.Timestamp, dt.date)):
        return o.isoformat()
    return str(o)


def jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default))


def clean(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def market_of(t: str) -> str:
    if t.endswith("-USD"):
        return "CRYPTO"
    if t.endswith(".HK"):
        return "HK"
    if t.endswith((".SS", ".SZ")):
        return "CN"
    return "US"


# ---------------------------------------------------------------- 股票池
def load_universe() -> dict[str, dict]:
    cache = DATA / "universe.json"
    uni: dict[str, dict] = {}

    def add(t, name, sector, tag):
        e = uni.setdefault(t, {"name": name, "market": market_of(t), "sector": sector or "", "tags": []})
        if tag not in e["tags"]:
            e["tags"].append(tag)

    try:
        sp = pd.read_csv(CFG["sources"]["sp500_csv"])
        for _, r in sp.iterrows():
            add(str(r["Symbol"]).replace(".", "-"), r.get("Security"), r.get("GICS Sector"), "SP500")
        log("标普500", len(sp))
    except Exception as e:  # noqa: BLE001
        log("标普500 名单获取失败：", e)

    try:
        tables = pd.read_html(CFG["sources"]["ndx_wiki"])
        ndx = next(t for t in tables if "Ticker" in t.columns or "Symbol" in t.columns)
        col = "Ticker" if "Ticker" in ndx.columns else "Symbol"
        name_col = "Company" if "Company" in ndx.columns else col
        for _, r in ndx.iterrows():
            add(str(r[col]).replace(".", "-"), r.get(name_col), "", "NDX100")
        log("纳斯达克100", len(ndx))
    except Exception as e:  # noqa: BLE001
        log("纳斯达克100 名单获取失败：", e)

    try:
        import akshare as ak
        cs = ak.index_stock_cons_csindex(symbol="000300")
        code_col = next(c for c in cs.columns if "代码" in c and "指数" not in c)
        name_col = next(c for c in cs.columns if "名称" in c and "指数" not in c)
        ex_col = next((c for c in cs.columns if "交易所" in c), None)
        for _, r in cs.iterrows():
            code = str(r[code_col]).zfill(6)
            ex = str(r[ex_col]) if ex_col else ""
            add(code + (".SS" if ("上海" in ex or code.startswith(("6", "9"))) else ".SZ"), r[name_col], "", "CSI300")
        log("沪深300", len(cs))
    except Exception as e:  # noqa: BLE001
        log("沪深300 名单获取失败：", e)

    hk = pd.read_csv(ROOT / "config" / "hk_major.csv", dtype=str)
    for _, r in hk.iterrows():
        add(r["ticker"], r["name"], r["sector"], "HK")
    for t in CFG["crypto"]:
        add(t, t, "Crypto", "CRYPTO")

    # 某个名单来源当天失败时，用上次缓存补齐
    if cache.exists():
        old = json.loads(cache.read_text())
        have = {tag for u in uni.values() for tag in u["tags"]}
        for src in ("SP500", "NDX100", "CSI300"):
            if src not in have:
                n = 0
                for t, v in old.items():
                    if src in v.get("tags", []):
                        add(t, v["name"], v.get("sector"), src)
                        n += 1
                log(f"{src} 使用缓存名单 {n} 只")
    jdump(uni, cache)
    return uni


# ---------------------------------------------------------------- 价格
def download_closes(tickers: list[str], period: str, interval: str, chunk: int = 150) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        for attempt in range(3):
            try:
                df = yf.download(part, period=period, interval=interval, auto_adjust=True,
                                 progress=False, threads=True, group_by="column")
                close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": part[0]})
                frames.append(close)
                break
            except Exception as e:  # noqa: BLE001
                log(f"下载失败 {period}/{interval} 第{attempt+1}次：", e)
                time.sleep(10 * (attempt + 1))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def tech_metrics(daily: pd.DataFrame, weekly: pd.DataFrame) -> dict[str, dict]:
    res = {}
    for t in daily.columns:
        s = daily[t].dropna()
        if s.empty:
            continue
        last = float(s.iloc[-1])
        w = weekly[t].dropna() if t in weekly.columns else pd.Series(dtype=float)
        res[t] = {
            "last": last,
            "last_date": s.index[-1].date().isoformat(),
            "chg1d": clean(last / s.iloc[-2] - 1) if len(s) >= 2 else None,
            "ret20": clean(last / s.iloc[-21] - 1) if len(s) >= 21 else None,
            "ret60": clean(last / s.iloc[-61] - 1) if len(s) >= 61 else None,
            "dd52": clean(last / s.max() - 1),
            "pct5y": clean((w <= last).mean()) if len(w) >= 52 else None,
        }
    return res


# ---------------------------------------------------------------- 基本面
def refresh_fundamentals(uni: dict, max_n: int) -> dict:
    path = DATA / "fundamentals.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    today = dt.date.today()
    stale = []
    for t, v in uni.items():
        if v["market"] == "CRYPTO":
            continue
        f = cache.get(t, {}).get("fetched")
        age = (today - dt.date.fromisoformat(f)).days if f else 10_000
        if age >= RULES["fund_ttl_days"]:
            stale.append((age, t))
    stale.sort(key=lambda x: -x[0])
    by_mkt: dict[str, list[str]] = {}
    for _, t in stale:
        by_mkt.setdefault(uni[t]["market"], []).append(t)
    todo: list[str] = []
    while any(by_mkt.values()) and len(todo) < max_n:
        for m in list(by_mkt):
            if by_mkt[m] and len(todo) < max_n:
                todo.append(by_mkt[m].pop(0))
    log(f"基本面：本次刷新 {len(todo)} 只（过期待刷 {len(stale)}）")

    fails = 0
    for i, t in enumerate(todo):
        try:
            info = yf.Ticker(t).info or {}
            rec = {k: info.get(k) for k in INFO_FIELDS}
            if rec.get("marketCap") is None and rec.get("trailingPE") is None:
                raise ValueError("空数据")
            rec["fetched"] = today.isoformat()
            cache[t] = rec
            fails = 0
        except Exception as e:  # noqa: BLE001
            fails += 1
            if fails in (5, 10):
                log(f"连续失败 {fails} 次（最近：{t} {e}），暂停 60 秒")
                time.sleep(60)
            if fails >= 15:
                log("连续失败过多，可能被限流，本次停止")
                break
        time.sleep(0.4)
        if (i + 1) % 100 == 0:
            log(f"  进度 {i+1}/{len(todo)}")
            jdump(cache, path)
    jdump(cache, path)
    return cache


def usd_rates(currencies: set[str]) -> dict[str, float]:
    """每 1 美元兑多少该货币。"""
    rates = {"USD": 1.0}
    need = sorted(c for c in currencies if c and c != "USD")
    closes = download_closes([f"{c}=X" for c in need], "5d", "1d") if need else pd.DataFrame()
    for c in need:
        col = f"{c}=X"
        if col in closes.columns and closes[col].dropna().size:
            rates[c] = float(closes[col].dropna().iloc[-1])
    return rates


def convert(x, frm, to, rates):
    if x is None or frm is None or to is None or frm == to:
        return x
    if frm not in rates or to not in rates:
        return None
    return x / rates[frm] * rates[to]


# ---------------------------------------------------------------- 筛选
def build_table(uni, tech, fund, rates) -> pd.DataFrame:
    rows = []
    for t, u in uni.items():
        m, f = tech.get(t, {}), fund.get(t, {})
        cur = f.get("currency")
        fcur = f.get("financialCurrency") or cur
        mcap = clean(f.get("marketCap"))
        if mcap is None and m.get("last"):
            sh = clean(f.get("sharesOutstanding")) or clean(f.get("impliedSharesOutstanding"))
            ni, eps = clean(f.get("netIncomeToCommon")), clean(f.get("trailingEps"))
            if not sh and ni and eps and eps > 0:
                sh = ni / eps
            mcap = sh * m["last"] if sh else None
        fcf = clean(f.get("freeCashflow"))
        fcf_c = convert(fcf, fcur, cur, rates) if fcf is not None else None
        fcf_yield = fcf_c / mcap if (fcf_c is not None and mcap) else None

        sector = f.get("sector") or u.get("sector") or ""
        industry = f.get("industry") or ""
        is_fin = sector in NO_FCF_SECTORS
        flags = []
        if is_fin:
            fcf_yield = None
            flags.append("金融/地产：不看现金流")
        elif fcf_yield is not None and fcf_yield > RULES["fcf_yield_suspect"]:
            flags.append(f"FCF收益率{fcf_yield*100:.0f}%，数据存疑")
            fcf_yield = None
        if sector in CYCLICAL_SECTORS or any(w in industry for w in CYCLICAL_WORDS):
            flags.append("周期股：当前利润可能处于高点")

        rev_g = clean(f.get("revenueGrowth"))
        g = min(max(rev_g, 0.0), 0.10) if rev_g is not None else None
        fpe = clean(f.get("forwardPE"))
        base_yield = (1 / fpe if fpe and fpe > 0 else None) if is_fin else fcf_yield
        rows.append({
            "ticker": t, "name": f.get("shortName") or u["name"], "market": u["market"],
            "sector": sector, "industry": industry, "index": "|".join(u["tags"]), "currency": cur,
            "price": m.get("last"), "price_date": m.get("last_date"),
            "chg1d": m.get("chg1d"), "ret20": m.get("ret20"), "ret60": m.get("ret60"),
            "dd52": m.get("dd52"), "pct5y": m.get("pct5y"),
            "mcap_usd": convert(mcap, cur, "USD", rates) if mcap else None,
            "pe": clean(f.get("trailingPE")), "fpe": fpe, "pb": clean(f.get("priceToBook")),
            "roe": clean(f.get("returnOnEquity")), "de": clean(f.get("debtToEquity")),
            "op_margin": clean(f.get("operatingMargins")), "fcf_yield": clean(fcf_yield),
            "div_yield": clean(f.get("trailingAnnualDividendYield")),
            "rev_growth": rev_g, "eps_growth": clean(f.get("earningsGrowth")),
            "analysts": clean(f.get("numberOfAnalystOpinions")), "target": clean(f.get("targetMeanPrice")),
            # 粗略隐含回报 = 现金流（或盈利）收益率 + 截在 0–10% 的营收增速。只用于排序，不是估值。
            "implied_return": clean(base_yield + g) if (base_yield is not None and g is not None) else None,
            "is_fin": is_fin, "flag": "；".join(flags), "fund_date": f.get("fetched"),
        })
    df = pd.DataFrame(rows)
    R = RULES

    def ok(col, cond):
        return df[col].notna() & cond(df[col])

    fin = df["is_fin"].astype(bool)
    hard = ok("roe", lambda s: s >= R["roe_min"]) & (ok("fcf_yield", lambda s: s > 0) | fin)
    soft_de = df["de"].isna() | (df["de"] <= R["de_max"]) | fin
    soft_om = df["op_margin"].isna() | (df["op_margin"] >= R["op_margin_min"]) | fin
    df["quality"] = hard & soft_de & soft_om

    c_fcf = ok("fcf_yield", lambda s: s >= R["fcf_yield_min"])
    c_pe = ok("fpe", lambda s: (s > 0) & (s <= R["fwd_pe_max"]))
    c_hist = ok("pct5y", lambda s: s <= R["pct5y_max"])
    c_pb = fin & ok("pb", lambda s: (s > 0) & (s <= R["pb_max_fin"]))
    df["cheap"] = c_fcf | c_pe | c_hist | c_pb
    df["cheap_why"] = [
        "、".join(x for x, b in (("FCF收益率≥5%", a), ("预期PE≤15", b_), ("5年价格低位", c), ("PB≤1.2", d)) if b)
        for a, b_, c, d in zip(c_fcf, c_pe, c_hist, c_pb)
    ]
    fell = ok("ret20", lambda s: s <= R["ret20_max"]) | ok("dd52", lambda s: s <= R["dd52_max"])
    # 盈利预期未恶化的代理：预期 PE 低于当前 PE，或最近利润同比为正
    not_worse = (df["fpe"].notna() & df["pe"].notna() & (df["fpe"] < df["pe"])) | ok("eps_growth", lambda s: s > 0)
    df["fallen"] = fell & not_worse
    return df


PICK_COLS = ["ticker", "name", "market", "sector", "industry", "currency", "price", "price_date",
             "chg1d", "ret20", "dd52", "pct5y", "pe", "fpe", "pb", "roe", "de", "op_margin",
             "fcf_yield", "div_yield", "rev_growth", "eps_growth", "implied_return", "cheap_why",
             "flag", "analysts", "target", "fund_date"]


def pick(df, mask, sort_col, asc, n, per_sector=2) -> list[dict]:
    sub = df[mask].sort_values(sort_col, ascending=asc, na_position="last")
    sub = sub.groupby(sub["sector"].replace("", "未知"), sort=False).head(per_sector)
    sub = sub.sort_values(sort_col, ascending=asc, na_position="last").head(n)
    return [{k: (clean(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
            for r in sub[PICK_COLS].to_dict("records")]


def pct(x):
    return "–" if x is None else f"{x*100:+.1f}%"


def num(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def write_report(day, screen, prices) -> str:
    st = screen["stats"]
    L = [f"# 全球质量价值筛选 {day}", "",
         f"股票池 {st['universe']} 只；有价格 {st['priced']}；有基本面 {st['with_fund']}；过质量门槛 {st['quality']}。", "",
         "## 一、质量 × 便宜（按粗略隐含回报排序，每行业最多 2 个）",
         "| 标的 | 市场 | 价格 | ROE | FCF收益率 | 预期PE | 5年分位 | 隐含回报 | 便宜依据 | 提示 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in screen["quality_cheap"]:
        L.append(f"| {r['name']} {r['ticker']} | {r['market']} | {num(r['price'],2)} | {pct(r['roe'])} | "
                 f"{pct(r['fcf_yield'])} | {num(r['fpe'])} | {num(r['pct5y'],2)} | {pct(r['implied_return'])} | "
                 f"{r['cheap_why']} | {r['flag'] or ''} |")
    L += ["", "## 二、错杀候选（质量过门槛、大跌、盈利预期未恶化）",
          "| 标的 | 市场 | 价格 | 20日 | 距52周高 | ROE | 预期PE | 现PE | 提示 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in screen["fallen"]:
        L.append(f"| {r['name']} {r['ticker']} | {r['market']} | {num(r['price'],2)} | {pct(r['ret20'])} | "
                 f"{pct(r['dd52'])} | {pct(r['roe'])} | {num(r['fpe'])} | {num(r['pe'])} | {r['flag'] or ''} |")
    L += ["", "## 汇率（每 1 外币兑人民币）", "、".join(f"{k} {v:.4f}" for k, v in prices["fx"].items() if v), "",
          "注：数据来自雅虎财经免费接口，港股/A股基本面有缺口；隐含回报只用于排序，不是估值结论。仅供研究，不构成投资建议。"]
    return "\n".join(L)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-fundamentals", type=int, default=400)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    day = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date().isoformat()
    log("运行日", day)
    uni = load_universe()
    tickers = sorted(uni)
    log("股票池", len(tickers))

    daily = download_closes(tickers, "1y", "1d")
    weekly = download_closes(tickers, "5y", "1wk")
    tech = tech_metrics(daily, weekly)
    fund = refresh_fundamentals(uni, args.max_fundamentals)
    rates = usd_rates({v.get("currency") for v in fund.values()} | {v.get("financialCurrency") for v in fund.values()})
    df = build_table(uni, tech, fund, rates)

    fxc = download_closes(list(CFG["fx"].values()), "5d", "1d")
    fx = {k: (round(float(fxc[v].dropna().iloc[-1]), 6) if v in fxc.columns and fxc[v].dropna().size else None)
          for k, v in CFG["fx"].items()}
    bmc = download_closes(list(CFG["benchmarks"].values()), "10d", "1d")
    bm = {}
    for k, t in CFG["benchmarks"].items():
        if t in bmc.columns and bmc[t].dropna().size >= 2:
            s = bmc[t].dropna()
            bm[k] = {"v": round(float(s.iloc[-1]), 2), "chg": round(float(s.iloc[-1] / s.iloc[-2] - 1), 4),
                     "date": s.index[-1].date().isoformat()}
    prices = {"date": day, "generated_utc": dt.datetime.utcnow().isoformat(timespec="minutes"),
              "source": "Yahoo Finance via yfinance", "fx_note": "每 1 外币兑人民币", "fx": fx,
              "benchmarks": bm,
              "quotes": {t: round(m["last"], 4) for t, m in tech.items()},
              "quote_dates": {t: m["last_date"] for t, m in tech.items()},
              "missing": sorted(set(uni) - set(tech))}

    screen = {
        "date": day, "rules": RULES,
        "stats": {"universe": len(df), "priced": int(df["price"].notna().sum()),
                  "with_fund": int(df["fund_date"].notna().sum()), "quality": int(df["quality"].sum())},
        "quality_cheap": pick(df, df["quality"] & df["cheap"], "implied_return", False, args.top),
        "fallen": pick(df, df["quality"] & df["fallen"], "ret20", True, args.top),
    }

    LATEST.mkdir(parents=True, exist_ok=True)
    jdump(prices, LATEST / "prices.json")
    jdump(screen, LATEST / "screen.json")
    (LATEST / "report.md").write_text(write_report(day, screen, prices))
    df.to_csv(LATEST / "metrics.csv", index=False)
    hist = HISTORY / day
    jdump(prices, hist / "prices.json")
    jdump(screen, hist / "screen.json")
    df.to_csv(hist / "metrics.csv.gz", index=False, compression="gzip")
    log("完成：", screen["stats"], "缺价格", len(prices["missing"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
