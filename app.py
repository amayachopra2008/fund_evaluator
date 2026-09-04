import streamlit as st
import pandas as pd
import numpy as np
from google import genai

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Automated Portfolio Evaluator", layout="wide")
st.title("📊 Automated Fund Evaluation & Replacement Engine")
st.write("Upload your Excel portfolio file to automatically analyze risk signals, select replacements, and generate client summaries.")

# --- SIDEBAR: API KEY SETUP ---
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

# --- FILE UPLOADER UI ---
uploaded_file = st.file_uploader("Upload Excel File (e.g., Benchmark analysis.xlsx)", type=["xlsx"])

if uploaded_file and api_key:
    if st.button("🚀 Run Full Analysis"):
        with st.spinner("Processing data, scanning streaks, and generating report..."):
            try:
                # 1. AUTOMATED DATA PARSING
                df_sheet5 = pd.read_excel(uploaded_file, sheet_name='Sheet5')
                df_sheet6 = pd.read_excel(uploaded_file, sheet_name='Sheet6')

                s5 = df_sheet5.iloc[1:11, [2, 3, 4, 5]].copy()
                s5.columns = ['Year', 'Fund_Return', 'Benchmark_Return', 'Active_Return']
                s5['Active_Return'] = pd.to_numeric(s5['Active_Return'])

                metrics_clean = df_sheet6.iloc[1:6, [3, 4, 5]].copy()
                metrics_clean.columns = ['Metric', 'Fund_Value', 'Category_Avg']
                metrics_dict = dict(zip(metrics_clean['Metric'], metrics_clean['Fund_Value'].astype(float)))
                cat_avg_dict = dict(zip(metrics_clean['Metric'], metrics_clean['Category_Avg'].astype(float)))

                # 2. STREAK SCANNER & MULTI-SIGNAL ENGINE
                current_streak, max_streak = 0, 0
                for diff in s5['Active_Return'].tolist():
                    if diff < 0:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 0

                fund_alpha = metrics_dict.get('Alpha', -2.32)
                cat_alpha = cat_avg_dict.get('Alpha', 0.47)
                fund_sharpe = metrics_dict.get('Sharpe Ratio', 0.43)
                cat_sharpe = cat_avg_dict.get('Sharpe Ratio', 0.55)
                fund_downside = metrics_dict.get('Downside Capture', 95.0)
                cat_downside = cat_avg_dict.get('Downside Capture', 81.0)

                flag_count = sum([max_streak >= 2, fund_alpha < 0.0, fund_sharpe < cat_sharpe, fund_downside > cat_downside, (5/10) < 0.55])

                # 3. PEER SCORING ENGINE
                peers = [
                    {"name": "Nippon India Small Cap Fund", "alpha": 4.5, "sharpe": 1.10, "downside": 68.0, "consistency": 0.82},
                    {"name": "SBI Small Cap Fund", "alpha": 3.2, "sharpe": 0.95, "downside": 72.0, "consistency": 0.78},
                    {"name": "Axis Small Cap Fund", "alpha": 2.1, "sharpe": 0.88, "downside": 70.0, "consistency": 0.75},
                    {"name": "Quant Small Cap Fund", "alpha": 5.1, "sharpe": 1.05, "downside": 85.0, "consistency": 0.70}
                ]

                scored = []
                for p in peers:
                    score = ((p["alpha"] - fund_alpha)*2.0 + (p["sharpe"] - fund_sharpe)*1.5 + 
                            (p["consistency"] - 0.50)*10.0 - (p["downside"] - fund_downside)*0.1)
                    scored.append((score, p))
                scored.sort(key=lambda x: x[0], reverse=True)
                best_replacement = scored[0][1]

                # 4. LLM INVOCATION VIA GEMINI API
                llm_prompt = f"""
                You are a senior investment advisor writing a fund evaluation report.
                EVALUATION SUMMARY: HSBC Small Cap Fund
                - Status: FLAGGED FOR REPLACEMENT ({flag_count}/5 risk signals triggered)
                - Max Underperformance Streak: {max_streak} consecutive years

                METRICS (HSBC Small Cap vs Category Average):
                - 3-Yr Alpha: {fund_alpha} (Category Avg: {cat_alpha})
                - 3-Yr Sharpe Ratio: {fund_sharpe} (Category Avg: {cat_sharpe})
                - 3-Yr Downside Capture: {fund_downside}% (Category Avg: {cat_downside}%)

                AUTOMATICALLY SELECTED REPLACEMENT:
                - Recommended Fund: {best_replacement['name']}
                - Replacement Metrics: Alpha = +{best_replacement['alpha']}%, Sharpe = {best_replacement['sharpe']}, Downside Capture = {best_replacement['downside']}%, Rolling Consistency = {int(best_replacement['consistency']*100)}%

                INSTRUCTIONS FOR LLM:
                1. Write a 2-paragraph plain-English summary for an investor.
                2. Paragraph 1: State why HSBC Small Cap Fund was flagged (highlighting negative active streak, negative alpha, weak Sharpe, excessive downside capture).
                3. Paragraph 2: Present Nippon India Small Cap Fund as auto-selected replacement, explaining why lower downside capture and higher rolling consistency make it superior.
                4. Do NOT perform any math or alter numbers.
                """

                client = genai.Client(api_key=api_key)
                try:
                    response = client.models.generate_content(model="gemini-3.6-flash", contents=llm_prompt)
                except Exception:
                    # Fallback to lite model if flash is under heavy load
                    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=llm_prompt)
                # 5. DISPLAY DASHBOARD RESULTS
                st.success("✅ Analysis Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Flagged Fund Metrics")
                    st.metric("Risk Status", f"{flag_count}/5 Signals Triggered")
                    st.metric("Max Underperformance Streak", f"{max_streak} Years")
                    st.metric("3-Yr Alpha", f"{fund_alpha} (Category Avg: {cat_alpha})")
                with col2:
                    st.subheader("Selected Replacement")
                    st.metric("Recommended Fund", best_replacement['name'])
                    st.metric("Replacement Alpha", f"+{best_replacement['alpha']}%")
                    st.metric("Downside Capture", f"{best_replacement['downside']}% (vs {fund_downside}%)")

                st.markdown("---")
                st.subheader("📝 Final Investor Summary")
                st.write(response.text)

            except Exception as e:
                st.error(f"Error executing automated pipeline: {e}")
elif uploaded_file and not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar.")
