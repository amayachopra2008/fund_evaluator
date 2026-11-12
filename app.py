import numpy as np
import pandas as pd
import streamlit as st

import engine as eng

st.set_page_config(page_title="Stock Early-Exit Engine", layout="wide")
st.title("📉 Stock Early-Exit Engine")
st.caption(
    "Upload or link a portfolio in almost any format. The app scores every stock for sustained underperformance, "
    "checks valuation (PE, PEG, 5-year average PE), flags concentration risk, then scans the market on its own "
    "for better same-sector replacements. Data: Yahoo Finance (free)."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API key (optional)", type="password",
                            help="Used only for the written investor summary and to read messy documents. "
                                 "Without it, a rule-based summary is shown.")
    rf = st.number_input("Risk-free rate (% p.a.)", 0.0, 15.0, eng.RISK_FREE * 100, 0.1) / 100
    sector_limit = st.number_input("Sector concentration limit (%)", 1.0, 100.0, eng.SECTOR_LIMIT_PCT, 0.5)
    stock_limit = st.number_input("Single-stock limit (%)", 1.0, 100.0, eng.STOCK_LIMIT_PCT, 0.5)
    with st.expander("Exit-signal thresholds"):
        th = dict(eng.THRESHOLDS)
        th["sharpe_min"] = st.number_input("Min Sharpe ratio", value=th["sharpe_min"], step=0.05)
        th["downside_max"] = st.number_input("Max downside capture (%)", value=th["downside_max"], step=1.0)
        th["drawdown_pct"] = st.number_input("Drawdown flag (% from 52w high)", value=th["drawdown_pct"], step=1.0)
        th["neg_year_streak"] = st.number_input("Negative-year streak", value=th["neg_year_streak"], step=1, min_value=1)
        th["neg_month_streak"] = st.number_input("Negative-month streak", value=th["neg_month_streak"], step=1, min_value=1)
        th["rel_6m_pct"] = st.number_input("6M relative return flag (pts)", value=th["rel_6m_pct"], step=1.0)
    max_cands = st.slider("Peers to scan per flagged stock", 8, 40, 20)

# ---------------------------------------------------------------- cached data wrappers
@st.cache_data(ttl=3600, show_spinner=False)
def cached_bench():
    return eng.fetch_history(eng.BENCHMARK)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_analysis(symbol: str, rf: float, th_json: str):
    import json
    return eng.analyze_stock(symbol, cached_bench(), rf, json.loads(th_json))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_resolve(name: str):
    return eng.resolve_ticker(name)


# ---------------------------------------------------------------- 1. input
st.subheader("1. Provide your portfolio")
tab_file, tab_url, tab_text = st.tabs(["Upload file", "Web link", "Paste text"])
uploaded = url = pasted = None
with tab_file:
    uploaded = st.file_uploader("Excel, CSV, TSV, TXT, PDF, Word (.docx), HTML",
                                type=["xlsx", "xlsm", "xls", "csv", "tsv", "txt", "pdf", "docx", "html", "htm"])
with tab_url:
    url = st.text_input("Link to a page, PDF, Excel, CSV or Word file")
with tab_text:
    pasted = st.text_area("Paste holdings (one per line, e.g. 'Reliance Industries, 9.5')", height=140)

if st.button("Read holdings", type="primary"):
    try:
        with st.spinner("Reading input..."):
            sheets, lines = eng.load_sources(uploaded, url.strip() if url else None, pasted)
            hold, note = eng.extract_holdings(sheets, lines)
            if hold.empty and api_key and lines:
                hold = eng.llm_extract_holdings("\n".join(lines), api_key)
                note = "extracted with Gemini"
        if hold.empty:
            st.error("Could not find any holdings. Try a different file, or paste them as text.")
        else:
            st.session_state["holdings"] = hold
            st.session_state["note"] = note
            st.session_state.pop("results", None)
    except Exception as e:
        st.error(f"Could not read the input: {e}")

# ---------------------------------------------------------------- 2. review
if "holdings" in st.session_state:
    st.subheader("2. Check the holdings")
    st.caption(f"Detected from: {st.session_state['note']}. Fix any name, weight or add rows before analysing. "
               "Leave weights blank to assume equal weights.")
    edited = st.data_editor(st.session_state["holdings"], num_rows="dynamic", use_container_width=True,
                            column_config={"input_name": "Stock name / ticker", "weight_pct": "Weight %"})

    if st.button("🚀 Analyse portfolio", type="primary"):
        import json
        th_json = json.dumps(th, sort_keys=True)
        edited = edited.dropna(subset=["input_name"])
        rows, unresolved = [], []
        prog = st.progress(0.0, text="Resolving tickers and fetching data...")
        for i, (_, r) in enumerate(edited.iterrows()):
            sym = cached_resolve(str(r["input_name"]))
            if not sym:
                unresolved.append(r["input_name"]); continue
            a = cached_analysis(sym, rf, th_json)
            a["input_name"], a["weight_pct"] = r["input_name"], r["weight_pct"]
            rows.append(a)
            prog.progress((i + 1) / len(edited), text=f"Analysed {sym}")
        prog.empty()

        ok = [a for a in rows if not a.get("error")]
        bad = [a for a in rows if a.get("error")]
        if not ok:
            st.error("No stock could be analysed. Yahoo Finance may be unreachable from this machine.")
            for b in bad:
                st.caption(f"{b['symbol']}: {b['error']}")
            st.stop()

        w = pd.Series([a["weight_pct"] for a in ok], dtype=float)
        equal = w.isna().all()
        if equal:
            w = pd.Series(100.0 / len(ok), index=range(len(ok)))
        else:
            w = w.fillna(0.0)
            w = w / w.sum() * 100
        for a, wt in zip(ok, w):
            a["weight_pct"] = round(float(wt), 2)

        port = pd.DataFrame([{"symbol": a["symbol"], "name": a["name"], "sector": a["sector"],
                              "weight_pct": a["weight_pct"]} for a in ok])
        sec_table, alerts = eng.concentration_report(port, sector_limit, stock_limit)
        st.session_state["results"] = dict(ok=ok, bad=bad, unresolved=unresolved, sec=sec_table, alerts=alerts,
                                           equal=equal, port=port, th=th)

# ---------------------------------------------------------------- 3. results
res = st.session_state.get("results")
if res:
    ok, bad, alerts = res["ok"], res["bad"], res["alerts"]
    st.subheader("3. Portfolio health")
    if res["equal"]:
        st.warning("No weights were found, so equal weights were assumed. Concentration flags depend on this.")
    for u in res["unresolved"]:
        st.caption(f"⚠️ Could not match '{u}' to a listed stock; skipped.")
    for b in bad:
        st.caption(f"⚠️ {b['symbol']}: {b['error']}")

    st.markdown("**Concentration risk**")
    if alerts:
        for a in alerts:
            st.error(a["msg"])
    else:
        st.success(f"No sector above {sector_limit:g}% and no stock above {stock_limit:g}%.")
    c1, c2 = st.columns([1, 1])
    c1.dataframe(res["sec"].rename(columns={"sector": "Sector", "weight_pct": "Weight %"}),
                 hide_index=True, use_container_width=True)

    table = pd.DataFrame([{
        "Stock": a["name"], "Weight %": a["weight_pct"], "Sector": a["sector"], "Status": a["status"],
        "PE": a["pe"], "5Y avg PE": a["avg_pe_5y"], "PEG": a["peg"], "Valuation": a["valuation"],
        "Alpha %": a["alpha"], "Sharpe": a["sharpe"], "Downside cap %": a["downside_capture"],
        "Drawdown %": a["current_drawdown"], "Red flags": len(a["flags"]),
    } for a in ok])
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("5Y avg PE is reconstructed from fiscal year-end price / diluted EPS; Yahoo's free feed usually has "
               "3-4 years of statements, so check the 'data points' figure in each stock's detail below.")

    flagged = [a for a in ok if a["status"] != "HOLD / MONITOR"]
    st.subheader("4. Stocks to exit and what to buy instead")
    if not flagged:
        st.success("No stock currently meets the early-exit criteria.")
    held = {a["symbol"] for a in ok}
    summaries = []
    for a in sorted(flagged, key=lambda x: -len(x["flags"])):
        with st.expander(f"{'🔴' if a['status'] == 'EXIT RECOMMENDED' else '🟠'} {a['name']}: {a['status']}",
                         expanded=True):
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("3Y alpha", f"{a['alpha']}%")
            m2.metric("Sharpe", a["sharpe"])
            m3.metric("Downside capture", f"{a['downside_capture']}%")
            m4.metric("Drawdown", f"{a['current_drawdown']}%")
            m5.metric("Negative-year streak", a["neg_year_streak"])
            st.markdown("**Red flags:** " + "; ".join(a["flags"]))
            st.markdown(f"**Valuation: {a['valuation']}**  \n" + "  \n".join(f"- {n}" for n in a["valuation_notes"]) +
                        f"  \n- 5Y avg PE built from {a['avg_pe_points']} data point(s); PEG source: {a['peg_source']}")
            with st.spinner(f"Scanning the market for peers of {a['name']}..."):
                reps, how, n_scanned = eng.find_replacements(a, held, cached_bench(), rf, res["th"], max_cands)
            if reps:
                st.markdown(f"**Replacement candidates** (scanned {n_scanned} peers via {how}; none were in your portfolio)")
                st.dataframe(pd.DataFrame([{
                    "Stock": r["name"], "Symbol": r["symbol"], "Industry": r["industry"], "Score": r["score"],
                    "Alpha %": r["alpha"], "Sharpe": r["sharpe"], "Downside cap %": r["downside_capture"],
                    "Drawdown %": r["current_drawdown"], "PE": r["pe"], "5Y avg PE": r["avg_pe_5y"],
                    "PEG": r["peg"], "Valuation": r["valuation"],
                } for r in reps]), hide_index=True, use_container_width=True)
                if any(al["level"] == "sector" and al["item"] == a["sector"] for al in alerts):
                    st.info(f"{a['sector']} is already over the sector limit; a same-sector swap keeps that "
                            "concentration unchanged.")
            else:
                st.warning(f"No suitable replacement found ({how}).")
            summ = eng.investor_summary(a, reps, api_key or None)
            st.markdown("**Investor summary**")
            st.write(summ)
            summaries.append((a["name"], summ))

    st.caption("Analysis based on historical data and free public feeds; not personalised investment advice.")
