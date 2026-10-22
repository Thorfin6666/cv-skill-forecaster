import streamlit as st
import pdfplumber
import json
import plotly.express as px
import pandas as pd
import google.generativeai as genai

# ==============================
# 1. Page Configuration
# ==============================
st.set_page_config(
    page_title="AI Skill Trend Forecaster",
    page_icon="💼",
    layout="wide"
)

st.title("💼 AI-Powered CV Skill Trend Forecaster")
st.caption(
    "Extract core technical skills from your resume and forecast their future market trajectory "
    "across global and Southeast Asian tech landscapes using Google Gemini."
)

# Sidebar: Configurations & File Upload
st.sidebar.header("Configuration & Upload")
gemini_key = st.sidebar.text_input(
    "Gemini API Key", 
    type="password", 
    help="Enter your Google AI Studio API key (starts with AIzaSy...)"
)
uploaded_file = st.sidebar.file_uploader(
    "Upload Your Resume / CV", 
    type=["pdf"],
    help="Please upload a text-selectable PDF resume."
)

# ==============================
# 2. PDF Text Extraction
# ==============================
def extract_text_from_pdf(file):
    """Extracts raw text content from an uploaded PDF file."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# ==============================
# 3. Gemini Forecasting Logic
# ==============================
def analyze_skills_with_gemini(cv_text, api_key):
    """Calls Gemini to extract skills and predict market trajectories."""
    genai.configure(api_key=api_key)

    # Use Google's recommended latest flash model
    target_model = "gemini-3.8-flash"

    model = genai.GenerativeModel(
        model_name=target_model,
        generation_config={"response_mime_type": "application/json"}
    )

    prompt = f"""
You are a senior tech recruiter and technology trend analyst specializing in global and Southeast Asian (Singapore, Malaysia, etc.) job markets.
Analyze the following resume text and perform two tasks:
1. Extract the 6 to 8 most critical hard technical skills (programming languages, frameworks, cloud platforms, databases, AI/data tools).
2. Evaluate the future trajectory and viability of each skill for the next 2-3 years based on tech evolution, hiring volume trends, and AI automation risks.

You must strictly return a valid JSON object matching the following structure without any markdown code wrappers:
{{
  "overall_summary": "A concise, objective summary of the candidate's technical profile competitiveness (1-2 sentences).",
  "skills": [
    {{
      "name": "Skill Name (e.g., Python, React, AWS)",
      "growth_score": 85,          // Projected market demand growth score (0-100)
      "ai_risk": 30,               // Risk of being automated or devalued by AI (0-100, lower is safer)
      "market_stage": "Rapid Growth", // One of: "Emerging", "Rapid Growth", "Mature & Stable", "Declining"
      "reason": "Brief rationale behind this forecast (under 25 words)."
    }}
  ],
  "recommended_upskills": [
    "Recommended high-value skill 1 to complement current stack",
    "Recommended high-value skill 2",
    "Recommended high-value skill 3"
  ]
}}

Resume text:
{cv_text[:4000]}
"""

    response = model.generate_content(prompt)
    clean_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(clean_text)

# ==============================
# 4. User Interface & Visualization
# ==============================
if uploaded_file and gemini_key:
    if st.button("🚀 Analyze Skills & Forecast Trends", type="primary"):
        with st.spinner("Gemini is analyzing your resume and modeling future tech trends..."):
            try:
                cv_text = extract_text_from_pdf(uploaded_file)
                if not cv_text.strip():
                    st.error("Could not extract readable text from this PDF. Please ensure it is not a scanned image.")
                    st.stop()
                    
                result = analyze_skills_with_gemini(cv_text, gemini_key)

                # Overall Profile Assessment
                st.info(f"💡 **Profile Assessment:** {result.get('overall_summary')}")

                skills_data = result.get("skills", [])
                df = pd.DataFrame(skills_data)

                col1, col2 = st.columns([3, 2])

                # Left Column: Quadrant Scatter Plot
                with col1:
                    st.markdown("#### 📊 Skill Viability Matrix (Strategic Quadrant)")
                    fig = px.scatter(
                        df, 
                        x="growth_score", 
                        y="ai_risk", 
                        text="name", 
                        color="market_stage",
                        size=[32] * len(df),
                        labels={
                            "growth_score": "Future Growth Potential (0 - 100)", 
                            "ai_risk": "AI Disruption / Automation Risk (0 - 100)",
                            "market_stage": "Market Phase"
                        },
                        title="Skill Positioning (Bottom-Right = High Growth & Safe from AI)"
                    )
                    fig.update_traces(textposition='top center')
                    fig.update_layout(
                        xaxis=dict(range=[0, 105], dtick=20),
                        yaxis=dict(range=[0, 105], dtick=20),
                        legend=dict(orientation="h", y=-0.2)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Right Column: Detailed Skill Breakdown
                with col2:
                    st.markdown("#### 🎯 Skill Trajectory Breakdown")
                    for s in skills_data:
                        with st.expander(f"{s['name']} — [{s['market_stage']}]"):
                            st.write(f"📈 **Growth Potential:** `{s['growth_score']}/100`")
                            st.write(f"🛡️ **AI Disruption Risk:** `{s['ai_risk']}/100`")
                            st.write(f"💬 **Market Insight:** {s['reason']}")

                # Bottom Section: Recommended Upskilling
                st.divider()
                st.markdown("#### 🌟 Recommended Upskilling Pathways")
                st.caption("Strategic capabilities to acquire next to maximize your market premium:")
                recs = result.get("recommended_upskills", [])
                cols = st.columns(len(recs))
                for i, rec in enumerate(recs):
                    cols[i].success(f"**Recommendation #{i+1}**\n\n{rec}")

            except Exception as e:
                st.error(f"Analysis failed: {e}")

elif not uploaded_file:
    st.info("👈 Please upload a PDF resume using the sidebar to begin.")
elif not gemini_key:
    st.warning("👈 Please enter your Gemini API Key in the sidebar.")