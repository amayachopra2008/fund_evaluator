"""
engine.py - analysis logic for the Stock Early-Exit Engine (no Streamlit imports,
so it can be unit-tested offline).

Data source: Yahoo Finance via the free `yfinance` package (no paid API needed).
"""
from __future__ import annotations

import io
import ipaddress
import json
import math
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# Defaults (all adjustable from the UI sidebar)
# ----------------------------------------------------------------------------
BENCHMARK = "^NSEI"            # Nifty 50
RISK_FREE = 0.065              # ~India 10Y G-Sec, annual
SECTOR_LIMIT_PCT = 15.0        # flag a sector above this weight
STOCK_LIMIT_PCT = 8.0          # flag a single stock above this weight

THRESHOLDS = {
    "sharpe_min": 0.55,          # from the original app (category average)
    "downside_max": 85.0,        # from the original app
    "alpha_min": 0.0,            # negative alpha = flag
    "neg_year_streak": 2,        # consecutive calendar years below benchmark
    "drawdown_pct": -25.0,       # current drawdown from 52-week high
    "neg_month_streak": 3,       # consecutive months below benchmark
    "rel_6m_pct": -10.0,         # 6-month return vs benchmark, in % points
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _to_num(x):
    """'12.5%', '1,234', '(5.2)', '₹ 900' -> float; otherwise NaN."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "-", "--", "n/a", "na"}:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[^\d.\-]", "", s)
    if s in {"", "-", ".", "-."}:
        return np.nan
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v


def _resample_alias(kind: str) -> str:
    """pandas >=2.2 uses 'ME'/'YE'; older versions use 'M'/'Y'."""
    new, old = {"month": ("ME", "M"), "year": ("YE", "Y")}[kind]
    try:
        pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-31"])).resample(new)
        return new
    except Exception:
        return old


def _clean_close(df: pd.DataFrame) -> pd.Series:
    s = df["Close"].dropna().astype(float)
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s[~s.index.duplicated(keep="last")]


# ----------------------------------------------------------------------------
# 1. UNIVERSAL INPUT READING
# ----------------------------------------------------------------------------
def _is_safe_url(url: str) -> bool:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            return False
        for info in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        return True
    except Exception:
        return False


def _read_pdf(fileobj) -> tuple[dict, list]:
    import pdfplumber
    sheets, lines = {}, []
    with pdfplumber.open(fileobj) as pdf:
        for pi, page in enumerate(pdf.pages):
            for ti, tbl in enumerate(page.extract_tables() or []):
                if tbl and len(tbl) > 1:
                    df = pd.DataFrame(tbl[1:], columns=[str(c) for c in tbl[0]])
                    sheets[f"pdf_p{pi + 1}_t{ti + 1}"] = df
            txt = page.extract_text() or ""
            lines.extend(txt.split("\n"))
    return sheets, lines


def _read_docx(fileobj) -> tuple[dict, list]:
    import docx
    d = docx.Document(fileobj)
    sheets, lines = {}, [p.text for p in d.paragraphs if p.text.strip()]
    for i, t in enumerate(d.tables):
        rows = [[c.text.strip() for c in r.cells] for r in t.rows]
        if len(rows) > 1:
            sheets[f"docx_t{i + 1}"] = pd.DataFrame(rows[1:], columns=rows[0])
    return sheets, lines


def _read_excel(fileobj) -> tuple[dict, list]:
    xls = pd.ExcelFile(fileobj)
    return {s: pd.read_excel(xls, sheet_name=s, header=None) for s in xls.sheet_names}, []


def _read_delimited_text(text: str) -> tuple[dict, list]:
    lines = text.splitlines()
    sheets = {}
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python", on_bad_lines="skip", header=None)
        if df.shape[1] >= 2 and len(df) >= 2:
            sheets["text_table"] = df
    except Exception:
        pass
    return sheets, lines


def _read_html(html: str) -> tuple[dict, list]:
    sheets = {}
    try:
        for i, t in enumerate(pd.read_html(io.StringIO(html))):
            sheets[f"web_table_{i + 1}"] = t
    except Exception:
        pass
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip()]
    return sheets, lines


def load_sources(uploaded=None, url: str | None = None, text: str | None = None):
    """Return (sheets: dict[str, DataFrame], raw_lines: list[str]) from any supported input."""
    if uploaded is not None:
        name = uploaded.name.lower()
        if name.endswith((".xlsx", ".xlsm", ".xls")):
            return _read_excel(uploaded)
        if name.endswith(".pdf"):
            return _read_pdf(uploaded)
        if name.endswith(".docx"):
            return _read_docx(uploaded)
        raw = uploaded.read()
        txt = raw.decode("utf-8", errors="ignore")
        if name.endswith((".html", ".htm")):
            return _read_html(txt)
        return _read_delimited_text(txt)

    if url:
        if not _is_safe_url(url):
            raise ValueError("That URL is not allowed (must be a public http/https address).")
        r = requests.get(url, headers=HEADERS, timeout=25)
        r.raise_for_status()
        if len(r.content) > 25 * 1024 * 1024:
            raise ValueError("File at that URL is larger than 25 MB.")
        ctype = r.headers.get("Content-Type", "").lower()
        low = url.lower().split("?")[0]
        buf = io.BytesIO(r.content)
        if low.endswith(".pdf") or "pdf" in ctype:
            return _read_pdf(buf)
        if low.endswith(".docx") or "wordprocessingml" in ctype:
            return _read_docx(buf)
        if low.endswith((".xlsx", ".xlsm", ".xls")) or "spreadsheet" in ctype or "excel" in ctype:
            return _read_excel(buf)
        if low.endswith((".csv", ".tsv", ".txt")) or "csv" in ctype or "text/plain" in ctype:
            return _read_delimited_text(r.text)
        return _read_html(r.text)

    if text and text.strip():
        return _read_delimited_text(text)
    return {}, []


# ----------------------------------------------------------------------------
# 2. HOLDING EXTRACTION
# ----------------------------------------------------------------------------
NAME_KEYS = ["symbol", "ticker", "scrip", "instrument", "stock", "security", "company", "holding", "equity", "name"]
WEIGHT_KEYS = ["weight", "allocation", "% of", "% to", "holding %", "holdings %", "portfolio %", "% nav",
               "%nav", "percent", "pct", "weightage", "% assets", "% holding"]
VALUE_KEYS = ["market value", "current value", "mkt value", "value", "amount", "invested", "valuation"]
QTY_KEYS = ["qty", "quantity", "shares", "units", "no. of"]
PRICE_KEYS = ["ltp", "cmp", "price", "close"]
SKIP_NAME_WORDS = ["total", "grand total", "cash", "net assets", "benchmark", "category", "average", "sub total",
                   "subtotal", "others", "equity & equity", "treasury", "tri-party", "tri party", "receivable",
                   "payable", "nan", "name", "symbol", "ticker"]
TICKER_STOPWORDS = {
    "NSE", "BSE", "PDF", "INR", "NAV", "TOTAL", "FUND", "LTD", "LIMITED", "THE", "AND", "FOR", "SIP", "ETF", "PE",
    "PEG", "AMC", "CAGR", "YTD", "SEBI", "AUM", "NIFTY", "SENSEX", "INDIA", "BANK", "CASH", "EQUITY", "DEBT",
    "SECTOR", "STOCK", "WEIGHT", "VALUE", "PRICE", "QTY", "PAGE", "DATE", "NAME", "SYMBOL", "TICKER", "ISIN",
    "MARKET", "CAP", "LARGE", "MID", "SMALL", "PORTFOLIO", "HOLDINGS", "RETURN", "RETURNS", "RISK", "TOP",
}


def _find_col(cols, keys, exclude=()):
    low = {c: str(c).strip().lower() for c in cols}
    for k in keys:
        for c, l in low.items():
            if k in l and not any(e in l for e in exclude):
                return c
    return None


def _promote_header(df: pd.DataFrame) -> pd.DataFrame:
    """If the real header row sits inside the first rows, promote it."""
    all_keys = NAME_KEYS + WEIGHT_KEYS + VALUE_KEYS + QTY_KEYS
    def hits(vals):
        return sum(any(k in str(v).strip().lower() for k in all_keys) for v in vals if str(v) != "nan")
    if hits(df.columns) >= 2:
        return df
    for i in range(min(15, len(df))):
        if hits(df.iloc[i].values) >= 2:
            out = df.iloc[i + 1:].copy()
            out.columns = [str(v).strip() for v in df.iloc[i].values]
            return out.reset_index(drop=True)
    return df


def _holdings_from_table(df: pd.DataFrame):
    df = _promote_header(df.copy())
    cols = list(df.columns)
    name_col = _find_col(cols, NAME_KEYS, exclude=("sector", "industry", "benchmark", "category", "isin"))
    if name_col is None:
        return None
    weight_col = _find_col(cols, WEIGHT_KEYS)
    value_col = _find_col(cols, VALUE_KEYS, exclude=("price", "gain", "pnl", "p&l", "return"))
    qty_col = _find_col(cols, QTY_KEYS)
    price_col = _find_col(cols, PRICE_KEYS, exclude=("avg", "buy", "purchase"))

    names = df[name_col].astype(str).str.strip()
    ok = ~names.str.lower().isin(SKIP_NAME_WORDS) & (names != "") & ~names.str.lower().str.contains(
        r"^(?:total|grand total|sub ?total)\b", regex=True)
    out = pd.DataFrame({"input_name": names[ok]})
    method = "none"
    if weight_col is not None:
        w = df.loc[ok, weight_col].map(_to_num)
        if w.notna().sum() >= max(1, int(0.5 * ok.sum())):
            out["weight_pct"] = w.values
            method = "weight column"
    if "weight_pct" not in out and value_col is not None:
        v = df.loc[ok, value_col].map(_to_num)
        if v.notna().sum() >= max(1, int(0.5 * ok.sum())):
            out["weight_pct"] = (v / v.sum() * 100).values
            method = "value column (converted to %)"
    if "weight_pct" not in out and qty_col is not None and price_col is not None:
        v = df.loc[ok, qty_col].map(_to_num) * df.loc[ok, price_col].map(_to_num)
        if v.notna().sum() >= max(1, int(0.5 * ok.sum())):
            out["weight_pct"] = (v / v.sum() * 100).values
            method = "qty x price (converted to %)"
    if "weight_pct" not in out:
        out["weight_pct"] = np.nan
    out = out[out["input_name"].str.len().between(2, 80)]
    return (out.reset_index(drop=True), method) if len(out) else None


def _holdings_from_text(lines: list[str]):
    rows = []
    pat = re.compile(r"^\s*(?:\d+[\.\)]\s*)?([A-Za-z][A-Za-z0-9&\.\-' ]{2,60}?)\s*[\-–:|,]?\s+(-?\d{1,3}(?:\.\d+)?)\s*%")
    for ln in lines:
        m = pat.match(ln)
        if m:
            nm = m.group(1).strip()
            if nm.lower() not in SKIP_NAME_WORDS and not any(w in nm.lower() for w in ("return", "total", "benchmark", "category")):
                rows.append((nm, float(m.group(2))))
    if rows:
        return pd.DataFrame(rows, columns=["input_name", "weight_pct"]), "text lines with % weights"
    # last resort: ticker-looking tokens (validated against Yahoo later; unresolved ones are dropped)
    found, seen = [], set()
    for ln in lines:
        for tok in re.findall(r"\b[A-Z][A-Z0-9&\-]{1,11}(?:\.(?:NS|BO))?\b", ln):
            base = tok.split(".")[0]
            if base in TICKER_STOPWORDS or base.isdigit() or tok in seen:
                continue
            seen.add(tok)
            found.append(tok)
    if found:
        return pd.DataFrame({"input_name": found[:40], "weight_pct": np.nan}), "ticker-like tokens in text"
    return None


def extract_holdings(sheets: dict, lines: list[str]):
    """Return (holdings_df[input_name, weight_pct], note)."""
    best, best_note = None, ""
    for name, df in sheets.items():
        if df is None or df.empty:
            continue
        res = _holdings_from_table(df)
        if res and (best is None or len(res[0]) > len(best)):
            best, best_note = res[0], f"table '{name}' ({res[1]})"
    if best is None:
        res = _holdings_from_text(lines)
        if res:
            best, best_note = res
    if best is None:
        return pd.DataFrame(columns=["input_name", "weight_pct"]), "No holdings found"
    best = best.groupby("input_name", as_index=False, sort=False).agg(
        weight_pct=("weight_pct", lambda s: s.sum(min_count=1)))
    return best, best_note


def llm_extract_holdings(text: str, api_key: str) -> pd.DataFrame:
    """Optional: ask Gemini to pull stock names + weights out of messy text."""
    prompt = (
        "Extract the stock holdings from the text below. Return ONLY a JSON list of objects with keys "
        '"name" (company name or exchange ticker exactly as written) and "weight_pct" '
        "(number or null). Do not invent holdings. Text:\n" + text[:15000]
    )
    raw = _gemini(prompt, api_key)
    m = re.search(r"\[.*\]", raw, re.S)
    data = json.loads(m.group(0)) if m else []
    df = pd.DataFrame([{"input_name": str(d.get("name", "")).strip(),
                        "weight_pct": _to_num(d.get("weight_pct"))} for d in data if d.get("name")])
    return df if len(df) else pd.DataFrame(columns=["input_name", "weight_pct"])


# ----------------------------------------------------------------------------
# 3. TICKER RESOLUTION
# ----------------------------------------------------------------------------
@lru_cache(maxsize=512)
def resolve_ticker(name: str):
    """Map a company name / bare symbol to a Yahoo ticker (prefers NSE, then BSE). Returns None if unresolved."""
    name = name.strip()
    if not name:
        return None
    if re.search(r"\.(NS|BO)$", name, re.I) or name.startswith("^"):
        return name.upper()
    quotes = []
    try:
        r = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                         params={"q": name, "quotesCount": 8, "newsCount": 0},
                         headers=HEADERS, timeout=10)
        quotes = r.json().get("quotes", [])
    except Exception:
        try:
            import yfinance as yf
            quotes = yf.Search(name, max_results=8).quotes
        except Exception:
            quotes = []
    eq = [q for q in quotes if q.get("quoteType") == "EQUITY" and str(q.get("symbol", "")).endswith((".NS", ".BO"))]
    for suffix in (".NS", ".BO"):
        for q in eq:
            if q["symbol"].endswith(suffix):
                return q["symbol"]
    if re.fullmatch(r"[A-Za-z0-9&\-]{2,15}", name):
        return name.upper() + ".NS"      # last guess; validated when prices are fetched
    return None


# ----------------------------------------------------------------------------
# 4. MARKET DATA + RISK METRICS
# ----------------------------------------------------------------------------
def fetch_history(symbol: str, period: str = "6y") -> pd.Series:
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No price history for {symbol}")
    return _clean_close(df)


def streak_counts(values) -> tuple[int, int]:
    """(max consecutive negatives, trailing consecutive negatives)."""
    cur = best = 0
    for v in values:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best, cur


def compute_risk_metrics(stock: pd.Series, bench: pd.Series, rf: float = RISK_FREE, years: int = 3):
    df = pd.concat([stock.rename("s"), bench.rename("b")], axis=1).dropna()
    if len(df) < 250:
        return None
    d3 = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]
    r = d3.pct_change().dropna()
    rf_d = (1 + rf) ** (1 / 252) - 1
    ex_s, ex_b = r["s"] - rf_d, r["b"] - rf_d
    var_b = ex_b.var()
    beta = ex_s.cov(ex_b) / var_b if var_b > 0 else np.nan
    alpha = (ex_s.mean() - beta * ex_b.mean()) * 252 * 100
    ann_ret = r["s"].mean() * 252
    vol = r["s"].std() * math.sqrt(252)
    sharpe = (ann_ret - rf) / vol if vol > 0 else np.nan

    m_alias, y_alias = _resample_alias("month"), _resample_alias("year")
    m = d3.resample(m_alias).last().pct_change().dropna()
    down = m[m["b"] < 0]
    downside = down["s"].mean() / down["b"].mean() * 100 if len(down) >= 3 else np.nan

    y = df.resample(y_alias).last().pct_change().dropna()
    yr_active = (y["s"] - y["b"]).tolist()
    y_max, y_trail = streak_counts(yr_active)

    m_all = df.resample(m_alias).last().pct_change().dropna()
    mo_max, mo_trail = streak_counts((m_all["s"] - m_all["b"]).tolist())

    last = df["s"].iloc[-1]
    hi52 = df["s"].iloc[-252:].max()
    cur_dd = (last / hi52 - 1) * 100
    max_dd = ((d3["s"] / d3["s"].cummax()) - 1).min() * 100
    n6 = min(126, len(df) - 1)
    rel6 = ((df["s"].iloc[-1] / df["s"].iloc[-1 - n6]) - (df["b"].iloc[-1] / df["b"].iloc[-1 - n6])) * 100

    f = lambda v, n=2: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), n)
    return {
        "alpha": f(alpha), "beta": f(beta), "sharpe": f(sharpe), "volatility": f(vol * 100),
        "downside_capture": f(downside, 1), "neg_year_streak": int(y_max), "neg_year_trailing": int(y_trail),
        "neg_month_streak": int(mo_trail), "current_drawdown": f(cur_dd, 1), "max_drawdown": f(max_dd, 1),
        "rel_6m": f(rel6, 1), "ann_return": f(ann_ret * 100, 1),
    }


def exit_assessment(m: dict, th: dict = THRESHOLDS):
    """Red flags + status. Rule of thumb carried over from the original app: >=2 flags -> exit review."""
    flags = []
    if m.get("alpha") is not None and m["alpha"] < th["alpha_min"]:
        flags.append(f"Negative alpha ({m['alpha']}% p.a. vs Nifty 50)")
    if m.get("sharpe") is not None and m["sharpe"] < th["sharpe_min"]:
        flags.append(f"Weak Sharpe ratio ({m['sharpe']} vs floor {th['sharpe_min']})")
    if m.get("downside_capture") is not None and m["downside_capture"] > th["downside_max"]:
        flags.append(f"High downside capture ({m['downside_capture']}% vs max {th['downside_max']}%)")
    if m["neg_year_streak"] >= th["neg_year_streak"]:
        flags.append(f"Lagged the benchmark {m['neg_year_streak']} calendar years in a row")
    if m["current_drawdown"] is not None and m["current_drawdown"] <= th["drawdown_pct"]:
        flags.append(f"Deep drawdown ({m['current_drawdown']}% from 52-week high)")
    if m["neg_month_streak"] >= th["neg_month_streak"]:
        flags.append(f"Beaten by the benchmark {m['neg_month_streak']} months running")
    if m["rel_6m"] is not None and m["rel_6m"] <= th["rel_6m_pct"]:
        flags.append(f"6-month return {abs(m['rel_6m'])} pts behind the benchmark")
    n = len(flags)
    status = "EXIT RECOMMENDED" if n >= 3 else "EXIT REVIEW" if n == 2 else "HOLD / MONITOR"
    return flags, status


# ----------------------------------------------------------------------------
# 5. VALUATION (PE, PEG, 5-year average PE)
# ----------------------------------------------------------------------------
def five_year_avg_pe(ticker_obj, close: pd.Series):
    """
    Approximation: price at each fiscal year-end / that year's diluted EPS, averaged over up to 5 years.
    Yahoo's free feed usually carries only ~4 annual statements, so the number of points is reported.
    Also returns EPS CAGR (used for PEG when Yahoo gives none).
    """
    try:
        inc = ticker_obj.income_stmt
        eps = None
        for label in ("Diluted EPS", "Basic EPS"):
            if label in inc.index:
                eps = inc.loc[label].dropna()
                break
        if eps is None or eps.empty:
            return None, 0, None
        eps = eps.sort_index().iloc[-5:]
        pes = []
        for dt, e in eps.items():
            dt = pd.Timestamp(dt).tz_localize(None) if pd.Timestamp(dt).tzinfo else pd.Timestamp(dt)
            if e > 0 and dt >= close.index[0]:
                px = close[:dt]
                if len(px):
                    pes.append(float(px.iloc[-1] / e))
        cagr = None
        pos = eps[eps > 0]
        if len(pos) >= 2:
            span = (pd.Timestamp(pos.index[-1]) - pd.Timestamp(pos.index[0])).days / 365.25
            if span >= 1:
                cagr = (pos.iloc[-1] / pos.iloc[0]) ** (1 / span) - 1
        return (float(np.mean(pes)) if pes else None), len(pes), cagr
    except Exception:
        return None, 0, None


def valuation_verdict(pe, avg_pe, peg):
    """Undervalued / Fairly priced / Overvalued from PE vs its own 5Y average and PEG."""
    if pe is None or pe <= 0:
        return {"verdict": "Not meaningful (loss-making / no PE)", "pe_vs_avg_pct": None, "notes": []}
    signals, notes, pe_vs = [], [], None
    if avg_pe and avg_pe > 0:
        ratio = pe / avg_pe
        pe_vs = round((ratio - 1) * 100, 1)
        if ratio <= 0.85:
            signals.append(-1); notes.append(f"PE {pe:.1f} is {abs(pe_vs)}% below its 5Y average ({avg_pe:.1f})")
        elif ratio >= 1.15:
            signals.append(1); notes.append(f"PE {pe:.1f} is {pe_vs}% above its 5Y average ({avg_pe:.1f})")
        else:
            signals.append(0); notes.append(f"PE {pe:.1f} is close to its 5Y average ({avg_pe:.1f})")
    if peg is not None and peg > 0:
        if peg < 1:
            signals.append(-1); notes.append(f"PEG {peg:.2f} < 1 (growth not fully priced in)")
        elif peg > 2:
            signals.append(1); notes.append(f"PEG {peg:.2f} > 2 (paying a lot for growth)")
        else:
            signals.append(0); notes.append(f"PEG {peg:.2f} is in the 1-2 fair range")
    elif peg is not None:
        notes.append("PEG not meaningful (earnings growth is negative)")
    if not signals:
        return {"verdict": "Insufficient data", "pe_vs_avg_pct": pe_vs, "notes": notes}
    total = sum(signals)
    verdict = "Undervalued" if total < 0 else "Overvalued" if total > 0 else "Fairly priced"
    if verdict == "Fairly priced" and len(set(signals)) > 1:
        notes.append("Signals conflict (one cheap, one expensive), so treated as fair")
    return {"verdict": verdict, "pe_vs_avg_pct": pe_vs, "notes": notes}


def analyze_stock(symbol: str, bench: pd.Series | None = None, rf: float = RISK_FREE, thresholds: dict = THRESHOLDS):
    """Full per-stock analysis. Never raises: returns {'symbol', 'error'} on failure."""
    try:
        import yfinance as yf
        if bench is None:
            bench = fetch_history(BENCHMARK)
        t = yf.Ticker(symbol)
        close = fetch_history(symbol)
        risk = compute_risk_metrics(close, bench, rf)
        if risk is None:
            return {"symbol": symbol, "error": "Less than ~1 year of price history"}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        price = info.get("currentPrice") or info.get("regularMarketPrice") or float(close.iloc[-1])
        pe = info.get("trailingPE")
        if pe is None and info.get("trailingEps"):
            pe = price / info["trailingEps"] if info["trailingEps"] > 0 else None
        avg_pe, n_pts, eps_cagr = five_year_avg_pe(t, close)

        peg, peg_src = info.get("trailingPegRatio") or info.get("pegRatio"), "Yahoo"
        if peg is None and pe and pe > 0:
            if eps_cagr and eps_cagr > 0:
                peg, peg_src = pe / (eps_cagr * 100), f"PE / EPS CAGR ({eps_cagr * 100:.1f}%)"
            elif info.get("earningsGrowth") and info["earningsGrowth"] > 0:
                peg, peg_src = pe / (info["earningsGrowth"] * 100), "PE / YoY earnings growth"
            else:
                peg, peg_src = (-1.0 if (eps_cagr is not None and eps_cagr <= 0) else None), "n/a"
        val = valuation_verdict(pe, avg_pe, peg if (peg is None or peg > 0) else (-1 if peg is not None else None))
        flags, status = exit_assessment(risk, thresholds)
        return {
            "symbol": symbol, "name": info.get("longName") or info.get("shortName") or symbol,
            "sector": info.get("sector") or "Unknown", "industry": info.get("industry") or "Unknown",
            "market_cap": info.get("marketCap"), "price": round(float(price), 2),
            "pe": round(pe, 2) if pe else None, "avg_pe_5y": round(avg_pe, 2) if avg_pe else None,
            "avg_pe_points": n_pts, "peg": round(peg, 2) if peg and peg > 0 else None, "peg_source": peg_src,
            **risk, "valuation": val["verdict"], "pe_vs_avg_pct": val["pe_vs_avg_pct"], "valuation_notes": val["notes"],
            "flags": flags, "status": status, "error": None,
        }
    except Exception as e:
        return {"symbol": symbol, "error": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------------------
# 6. CONCENTRATION RISK
# ----------------------------------------------------------------------------
def concentration_report(df: pd.DataFrame, sector_limit=SECTOR_LIMIT_PCT, stock_limit=STOCK_LIMIT_PCT):
    """df needs columns: symbol, name, sector, weight_pct (sums to ~100). Returns (sector_table, alerts)."""
    sec = (df.groupby("sector", as_index=False)["weight_pct"].sum()
           .sort_values("weight_pct", ascending=False).reset_index(drop=True))
    alerts = []
    for _, r in sec.iterrows():
        if r["weight_pct"] > sector_limit and r["sector"] != "Unknown":
            alerts.append({"level": "sector", "item": r["sector"], "weight": round(r["weight_pct"], 2),
                           "limit": sector_limit,
                           "msg": f"Sector '{r['sector']}' is {r['weight_pct']:.1f}% of the portfolio (limit {sector_limit:g}%)"})
    for _, r in df.sort_values("weight_pct", ascending=False).iterrows():
        if r["weight_pct"] > stock_limit:
            alerts.append({"level": "stock", "item": r["symbol"], "weight": round(r["weight_pct"], 2),
                           "limit": stock_limit,
                           "msg": f"{r['name']} is {r['weight_pct']:.1f}% of the portfolio (limit {stock_limit:g}%)"})
    return sec, alerts


# ----------------------------------------------------------------------------
# 7. AUTONOMOUS PEER DISCOVERY
# ----------------------------------------------------------------------------
def discover_peer_symbols(target: dict, exclude: set, max_candidates: int = 20):
    """
    Scan Yahoo's screener for same-sector Indian stocks of similar size (0.25x-4x market cap),
    then fall back to Yahoo's 'similar stocks' recommendations. Nothing is hard-coded.
    Returns (symbols, method_note).
    """
    notes, pool = [], {}
    sector, mcap = target.get("sector"), target.get("market_cap")
    try:
        import yfinance as yf
        from yfinance import EquityQuery
        lo, hi = (mcap * 0.25, mcap * 4) if mcap else (None, None)
        for use_exchange in (True, False):
            conds = [EquityQuery("eq", ["region", "in"]), EquityQuery("eq", ["sector", sector])]
            if use_exchange:
                conds.append(EquityQuery("eq", ["exchange", "NSI"]))
            if lo:
                conds.append(EquityQuery("btwn", ["intradaymarketcap", lo, hi]))
            res = yf.screen(EquityQuery("and", conds), size=100, sortField="intradaymarketcap", sortAsc=False)
            for q in res.get("quotes", []):
                sym = q.get("symbol", "")
                if sym.endswith((".NS", ".BO")):
                    pool[sym] = q.get("marketCap") or 0
            if pool:
                notes.append("Yahoo Finance equity screener (same sector, 0.25x-4x market cap)")
                break
    except Exception as e:
        notes.append(f"screener unavailable ({type(e).__name__})")
    if len(pool) < 5:
        try:
            r = requests.get(f"https://query2.finance.yahoo.com/v6/finance/recommendationsbysymbol/{target['symbol']}",
                             headers=HEADERS, timeout=10)
            for item in r.json()["finance"]["result"][0]["recommendedSymbols"]:
                if item["symbol"].endswith((".NS", ".BO")):
                    pool.setdefault(item["symbol"], 0)
            notes.append("Yahoo 'similar stocks' recommendations")
        except Exception:
            pass
    base = lambda s: s.split(".")[0]
    ex = {base(s) for s in exclude}
    cands = {s: m for s, m in pool.items() if base(s) not in ex and base(s) != base(target["symbol"])}
    # keep one listing per company (prefer NSE)
    uniq = {}
    for s, m in cands.items():
        if base(s) not in uniq or s.endswith(".NS"):
            uniq[base(s)] = (s, m)
    ordered = sorted(uniq.values(), key=lambda x: abs(math.log((x[1] or mcap or 1) / (mcap or x[1] or 1))) if (mcap and x[1]) else 0)
    return [s for s, _ in ordered[:max_candidates]], "; ".join(notes) or "no source reachable"


def rank_replacements(target: dict, candidates: list[dict], top_n: int = 5):
    """Score candidates relative to the underperformer. Higher is better. Healthy, fairly valued names only."""
    val_bonus = {"Undervalued": 2.0, "Fairly priced": 1.0, "Overvalued": -2.0}
    out = []
    for c in candidates:
        if c.get("error") or c["sector"] != target["sector"]:
            continue
        if c["valuation"].startswith("Not meaningful") or len(c["flags"]) >= 2:
            continue
        g = lambda d, k, dflt: d[k] if d.get(k) is not None else dflt
        score = (
            (g(c, "alpha", 0) - g(target, "alpha", 0)) * 0.10
            + (g(c, "sharpe", 0) - g(target, "sharpe", 0)) * 1.5
            - (g(c, "downside_capture", 100) - g(target, "downside_capture", 100)) * 0.02
            + (g(c, "current_drawdown", -50) - g(target, "current_drawdown", -50)) * 0.05
            + val_bonus.get(c["valuation"], 0)
            + (0.5 if c["industry"] == target["industry"] else 0)
            - len(c["flags"]) * 0.5
        )
        out.append({**c, "score": round(score, 2)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_n]


def find_replacements(target: dict, exclude: set, bench: pd.Series, rf=RISK_FREE, thresholds=THRESHOLDS,
                      max_candidates=20, top_n=5):
    syms, how = discover_peer_symbols(target, exclude, max_candidates)
    if not syms:
        return [], how, 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(lambda s: analyze_stock(s, bench, rf, thresholds), syms))
    return rank_replacements(target, res, top_n), how, len(syms)


# ----------------------------------------------------------------------------
# 8. SUMMARY (Gemini optional, deterministic fallback)
# ----------------------------------------------------------------------------
def _gemini(prompt: str, api_key: str) -> str:
    from google import genai
    client = genai.Client(api_key=api_key)
    last = None
    for model in ("gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"):
        try:
            return client.models.generate_content(model=model, contents=prompt).text
        except Exception as e:
            last = e
    raise last


def facts_block(stock: dict, replacements: list[dict]) -> str:
    lines = [
        f"STOCK: {stock['name']} ({stock['symbol']}), sector {stock['sector']}, status {stock['status']}",
        f"Weight in portfolio: {stock.get('weight_pct', 'n/a')}%",
        f"3Y alpha {stock['alpha']}%, Sharpe {stock['sharpe']}, downside capture {stock['downside_capture']}%, "
        f"current drawdown {stock['current_drawdown']}%, negative-year streak {stock['neg_year_streak']}",
        f"PE {stock['pe']}, 5Y avg PE {stock['avg_pe_5y']} ({stock['avg_pe_points']} data points), PEG {stock['peg']}, "
        f"valuation: {stock['valuation']}",
        "Red flags: " + ("; ".join(stock["flags"]) or "none"),
    ]
    for i, r in enumerate(replacements, 1):
        lines.append(f"REPLACEMENT {i}: {r['name']} ({r['symbol']}) alpha {r['alpha']}%, Sharpe {r['sharpe']}, "
                     f"downside capture {r['downside_capture']}%, drawdown {r['current_drawdown']}%, "
                     f"PE {r['pe']} vs 5Y avg {r['avg_pe_5y']}, PEG {r['peg']}, {r['valuation']}")
    return "\n".join(lines)


def investor_summary(stock: dict, replacements: list[dict], api_key: str | None = None) -> str:
    facts = facts_block(stock, replacements)
    if api_key:
        try:
            return _gemini(
                "You are a senior equity research analyst. Write a SHORT investor summary (max 150 words, plain "
                "language, 3 short paragraphs: diagnosis, valuation, action). Use ONLY the facts below, do not "
                "invent numbers, and note this is analysis, not personalised advice.\n\n" + facts, api_key)
        except Exception:
            pass
    s = stock
    top = replacements[0] if replacements else None
    txt = (f"**{s['name']}** is flagged **{s['status']}** with {len(s['flags'])} red flag(s): "
           f"{'; '.join(s['flags']) or 'none'}. On valuation it looks **{s['valuation'].lower()}** "
           f"(PE {s['pe']}, 5Y avg PE {s['avg_pe_5y']}, PEG {s['peg']}). ")
    if top:
        txt += (f"The best same-sector replacement found by the scan is **{top['name']}** "
                f"(alpha {top['alpha']}%, Sharpe {top['sharpe']}, {top['valuation'].lower()}). ")
    txt += "This is data-driven analysis, not personalised investment advice."
    return txt
