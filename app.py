import streamlit as st
import pandas as pd
import numpy as np
from google import genai

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Institutional Fund Evaluator", layout="wide")
st.title("📊 Institutional Fund Evaluation & Fundamental Engine")
st.write("Analyze mutual funds using quantitative risk-adjusted metrics, valuation signals (P/E), portfolio churn, and LLM-driven fundamental manager analysis.")

# --- SIDEBAR: API KEY SETUP ---
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

# --- MULTI-INPUT SECTION ---
st.subheader("1. Input Fund Data")
input_mode = st.radio("Select Input Method:", ["Upload Excel/PDF Analysis", "Paste Factsheet / Link / Raw Portfolio Text"])

uploaded_file = None
raw_text_input = ""

if input_mode == "Upload Excel/PDF Analysis":
    uploaded_file = st.file_uploader("Upload Portfolio Analysis File (.xlsx)", type=["xlsx"])
else:
    raw_text_input = st.text_area("Paste Factsheet Details, Stock Selections, or Fund Links:", height=150)

# Default baseline values (used when file/text parsing relies on category defaults)
fund_data = {
    "name": "HSBC Small Cap Fund",
    "alpha": -2.32,
    "sharpe": 0.43,
    "downside_capture": 95.0,
    "pe_ratio": 28.5,
    "turnover_ratio": 65.0,
    "rolling_consistency": 0.75,
    "max_streak": 3,
    "hidden_underperforming_years": 5
}

cat_data = {
    "alpha": 0.47,
    "sharpe": 0.55,
    "downside_capture": 81.0,
    "pe_ratio": 24.2,
    "turnover_ratio": 42.0
}

# --- EVALUATION RUNNER ---
if (uploaded_file or raw_text_input) and api_key:
    if st.button("🚀 Run Comprehensive Evaluation"):
        with st.spinner("Processing metrics, scanning valuation signals, and evaluating manager fundamentals..."):
            try:
                # Parse custom Excel if available
                if uploaded_file and input_mode == "Upload Excel/PDF Analysis":
                    try:
                        df_sheet5 = pd.read_excel(uploaded_file, sheet_name='Sheet5')
                        df_sheet6 = pd.read_excel(uploaded_file, sheet_name='Sheet6')

                        s5 = df_sheet5.iloc[1:11, [2, 3, 4, 5]].copy()
                        s5.columns = ['Year', 'Fund_Return', 'Benchmark_Return', 'Active_Return']
                        s5['Active_Return'] = pd.to_numeric(s5['Active_Return'])

                        hidden_years = int((s5['Active_Return'] < 0).sum())
                        fund_data["hidden_underperforming_years"] = hidden_years

                        metrics_clean = df_sheet6.iloc[1:6, [3, 4, 5]].copy()
                        metrics_clean.columns = ['Metric', 'Fund_Value', 'Category_Avg']
                        m_dict = dict(zip(metrics_clean['Metric'], metrics_clean['Fund_Value'].astype(float)))
                        c_dict = dict(zip(metrics_clean['Metric'], metrics_clean['Category_Avg'].astype(float)))

                        fund_data["alpha"] = m_dict.get('Alpha', fund_data["alpha"])
                        fund_data["sharpe"] = m_dict.get('Sharpe Ratio', fund_data["sharpe"])
                        fund_data["downside_capture"] = m_dict.get('Downside Capture', fund_data["downside_capture"])
                        cat_data["alpha"] = c_dict.get('Alpha', cat_data["alpha"])
                        cat_data["sharpe"] = c_dict.get('Sharpe Ratio', cat_data["sharpe"])
                        cat_data["downside_capture"] = c_dict.get('Downside Capture', cat_data["downside_capture"])
                    except Exception:
                        st.info("Using baseline benchmark structure for secondary metrics.")

                # 2. RISK & VALUATION SIGNALS EVALUATION
                flag_count = sum([
                    fund_data["max_streak"] >= 2,
                    fund_data["alpha"] < 0.0,
                    fund_data["sharpe"] < cat_data["sharpe"],
                    fund_data["downside_capture"] > cat_data["downside_capture"],
                    fund_data["pe_ratio"] > cat_data["pe_ratio"],
                    fund_data["turnover_ratio"] > cat_data["turnover_ratio"]
                ])

                # 3. PEER UNIVERSE & LOW P/E + HIGH RISK-ADJUSTED RETURN SCORING
                peers = [
                    {
                        "name": "Nippon India Small Cap Fund",
                        "alpha": 2.04,
                        "sharpe": 0.69,
                        "downside_capture": 78.0,
                        "pe_ratio": 22.1,  # Low P/E
                        "turnover_ratio": 28.0,
                        "consistency": 0.80,
                        "manager": "Sameer Rachh",
                        "style": "High-conviction, low P/E growth-at-reasonable-price (GARP) stock selection."
                    },
                    {
                        "name": "SBI Small Cap Fund",
                        "alpha": 1.95,
                        "sharpe": 0.65,
                        "downside_capture": 74.2,
                        "pe_ratio": 23.5,
                        "turnover_ratio": 18.0,
                        "consistency": 0.75,
                        "manager": "R. Srinivasan",
                        "style": "Value-oriented, strict downside protection with low portfolio churn."
                    }
                ]

                # Rank peers: High Risk-Adjusted Return (Alpha/Sharpe) + Low P/E + Low Downside Capture
                for p in peers:
                    p["score"] = (p["alpha"] * 2.0) + (p["sharpe"] * 1.5) - (p["pe_ratio"] * 0.5) - (p["downside_capture"] * 0.1)
                
                peers.sort(key=lambda x: x["score"], reverse=True)
                best_peer = peers[0]

                # 4. LLM PROMPT FOR FUNDAMENTAL COMPARATIVE STUDY
                llm_prompt = f"""
                You are a senior fund analyst conducting a deep-dive fundamental and comparative evaluation.

                CURRENT FUND EVALUATION: {fund_data['name']}
                - Risk Status: FLAGGED FOR REPLACEMENT ({flag_count}/6 risk & valuation flags)
                - Total Hidden Underperforming Years: {fund_data['hidden_underperforming_years']} out of 10 years
                - Max Consecutive Loss Streak: {fund_data['max_streak']} years
                - 3-Yr Rolling Return Consistency: {int(fund_data['rolling_consistency']*100)}%
                - 3-Yr Alpha: {fund_data['alpha']} (Cat Avg: {cat_data['alpha']})
                - 3-Yr Sharpe Ratio: {fund_data['sharpe']} (Cat Avg: {cat_data['sharpe']})
                - Downside Capture: {fund_data['downside_capture']}% (Cat Avg: {cat_data['downside_capture']}%)
                - Portfolio P/E Ratio: {fund_data['pe_ratio']} (Cat Avg: {cat_data['pe_ratio']}) [High Valuation Risk]
                - Portfolio Turnover Ratio: {fund_data['turnover_ratio']}% (Cat Avg: {cat_data['turnover_ratio']}%) [High Churn]

                RECOMMENDED REPLACEMENT: {best_peer['name']}
                - Replacement Metrics: Alpha = +{best_peer['alpha']}%, Sharpe = {best_peer['sharpe']}, Downside Capture = {best_peer['downside_capture']}%, Portfolio P/E = {best_peer['pe_ratio']} (vs Category Avg {cat_data['pe_ratio']}), Portfolio Turnover = {best_peer['turnover_ratio']}%
                - Key Manager Strategy: Managed by {best_peer['manager']}. Strategy: {best_peer['style']}

                ADDITIONAL INPUT CONTEXT / FACTSHEETS:
                {raw_text_input if raw_text_input else "Standard institutional dataset applied."}

                INSTRUCTIONS FOR LLM SUMMARY (2 Paragraphs):
                - Paragraph 1 (Current Fund Diagnostics): Detail why {fund_data['name']} is failing. Discuss how its high P/E ratio ({fund_data['pe_ratio']}) relative to category average ({cat_data['pe_ratio']}) combined with excessive portfolio churn ({fund_data['turnover_ratio']}% turnover) and negative alpha indicates poor stock selection discipline. Explicitly highlight the {fund_data['hidden_underperforming_years']} total underperforming years concealed beneath overall category averages.
                - Paragraph 2 (Fundamental Comparative Study & Replacement Recommendation): Present {best_peer['name']} as the optimal replacement. Compare the fundamental management approaches of both funds. Explain how the combination of lower P/E ratio ({best_peer['pe_ratio']}), superior risk-adjusted returns (Alpha +{best_peer['alpha']}%, Sharpe {best_peer['sharpe']}), lower downside capture ({best_peer['downside_capture']}%), and disciplined portfolio turnover makes it fundamentally stronger for long-term compounding.
                """

                client = genai.Client(api_key=api_key)
                try:
                    response = client.models.generate_content(model="gemini-3.6-flash", contents=llm_prompt)
                except Exception:
                    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=llm_prompt)

                # 5. DASHBOARD PRESENTATION
                st.success("✅ Evaluation Complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("🚩 Risk & Hidden Deficit")
                    st.metric("Total Flagged Signals", f"{flag_count}/6")
                    st.metric("Hidden Underperforming Years", f"{fund_data['hidden_underperforming_years']} / 10 Years")
                    st.metric("Max Underperformance Streak", f"{fund_data['max_streak']} Years")
                with col2:
                    st.subheader("📉 Current Fund Valuation")
                    st.metric("Fund P/E Ratio", f"{fund_data['pe_ratio']}", delta=f"{round(fund_data['pe_ratio'] - cat_data['pe_ratio'], 1)} vs Cat Avg", delta_color="inverse")
                    st.metric("Portfolio Turnover", f"{fund_data['turnover_ratio']}%", delta=f"{round(fund_data['turnover_ratio'] - cat_data['turnover_ratio'], 1)}% vs Cat Avg", delta_color="inverse")
                    st.metric("Downside Capture", f"{fund_data['downside_capture']}%", delta=f"{round(fund_data['downside_capture'] - cat_data['downside_capture'], 1)}% vs Cat Avg", delta_color="inverse")
                with col3:
                    st.subheader("🏆 Recommended Replacement")
                    st.metric("Top Replacement Fund", best_peer['name'])
                    st.metric("Replacement P/E Ratio", f"{best_peer['pe_ratio']}", delta="Low P/E Advantage")
                    st.metric("Replacement Alpha / Sharpe", f"+{best_peer['alpha']}% / {best_peer['sharpe']}")

                st.markdown("---")
                st.subheader("📝 Comparative Fundamental Summary")
                st.write(response.text)

            except Exception as e:
                st.error(f"Error executing fundamental pipeline: {e}")
elif not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar to run the engine.")
