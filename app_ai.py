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
    page_title="AI Skill Trajectory & Viability Forecaster",
    page_icon="💼",
    layout="wide"
)

st.title("💼 AI-Powered CV Skill Trajectory Forecaster")
st.caption(
    "Extract core technical skills from your resume and forecast their future market viability "
    "using an explainable multi-dimensional scoring rubric powered by Google Gemini and deterministic Python calculations."
)

# Sidebar: Configurations & Secrets Fallback
st.sidebar.header("Configuration & Upload")

# Safely attempt to read secrets (prevents StreamlitSecretNotFoundError locally)
try:
    default_key = st.secrets.get("GEMINI_KEY", "")
except Exception:
    default_key = ""

gemini_key = st.sidebar.text_input(
    "Gemini API Key", 
    value=default_key,
    type="password", 
    help="Enter your Google AI Studio API key (starts with AIzaSy...)"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Your Resume / CV", 
    type=["pdf"],
    help="Upload a text-selectable PDF resume."
)

# Sidebar: Mathematical Methodology
with st.sidebar.expander("📐 Deterministic Scoring Formulas"):
    st.markdown("""
    **Growth Potential Formula:**
    $$\\text{Growth} = 0.4 S_{\\text{demand}} + 0.3 S_{\\text{ecosystem}} + 0.3 S_{\\text{premium}}$$
    
    **AI Disruption Risk Formula:**
    $$\\text{AI Risk} = 0.4 R_{\\text{standard}} + 0.4 R_{\\text{tools}} - 0.2 R_{\\text{judgment}}$$
    """)

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
# 3. Hybrid AI + Python Forecasting Engine
# ==============================
def analyze_skills_with_gemini(cv_text, api_key):
    """
    Hybrid Architecture:
    1. Gemini evaluates qualitative sub-dimension scores (0-100).
    2. Python strictly executes deterministic mathematical formulas.
    """
    genai.configure(api_key=api_key)

    target_model = "gemini-3.8-flash"

    model = genai.GenerativeModel(
        model_name=target_model,
        generation_config={"response_mime_type": "application/json"}
    )

    prompt = f"""
You are a senior tech talent analyst and workforce economist specializing in global and Southeast Asian job markets.
Analyze the following resume text and execute two objectives:

1. Extract the 6 to 8 most critical hard technical skills (programming languages, frameworks, cloud/DevOps, data/AI tools, specialized software).
2. For each skill, evaluate the sub-dimension ratings (0 to 100) based on market dynamics for the next 2-3 years:

SUB-DIMENSIONS TO RATE (0 - 100):
- demand_trajectory: Hiring volume momentum & YoY job postings.
- ecosystem_alignment: Modern technology stack relevance.
- salary_premium: Scarcity and compensation leverage.
- task_standardization: How routine and syntax-bound the tasks are.
- ai_tool_maturity: Availability of mature generative AI/copilot tools.
- human_judgment_barrier: Need for physical constraints, architecture, cross-domain judgment.

Return strictly a valid JSON object matching this schema without any markdown wrapping:
{{
  "overall_summary": "Objective synthesis of the candidate's technical profile competitiveness (1-2 sentences).",
  "skills": [
    {{
      "name": "Skill Name",
      "market_stage": "Rapid Growth", // One of: "Emerging", "Rapid Growth", "Mature & Stable", "Declining"
      "growth_breakdown": {{
        "demand_trajectory": 85,
        "ecosystem_alignment": 80,
        "salary_premium": 75
      }},
      "ai_risk_breakdown": {{
        "task_standardization": 45,
        "ai_tool_maturity": 60,
        "human_judgment_barrier": 65
      }},
      "reason": "Concise justification for this rating (max 25 words)."
    }}
  ],
  "recommended_upskills": [
    "High-leverage skill recommendation 1 to future-proof current stack",
    "High-leverage skill recommendation 2",
    "High-leverage skill recommendation 3"
  ]
}}

Resume Content:
{cv_text[:4000]}
"""

    response = model.generate_content(prompt)
    clean_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
    data = json.loads(clean_text)

    # ==========================================
    # 🔥 Deterministic Mathematical Calculation in Python
    # ==========================================
    for s in data.get("skills", []):
        gb = s.get("growth_breakdown", {})
        ab = s.get("ai_risk_breakdown", {})

        # Formula 1: Growth Potential = 0.40 * Demand + 0.30 * Ecosystem + 0.30 * Premium
        real_growth = (
            0.40 * gb.get("demand_trajectory", 50) +
            0.30 * gb.get("ecosystem_alignment", 50) +
            0.30 * gb.get("salary_premium", 50)
        )
        s["growth_score"] = int(round(max(0, min(100, real_growth))))

        # Formula 2: AI Disruption Risk = 0.40 * Standard + 0.40 * ToolMaturity - 0.20 * HumanBarrier
        real_risk = (
            0.40 * ab.get("task_standardization", 50) +
            0.40 * ab.get("ai_tool_maturity", 50) -
            0.20 * ab.get("human_judgment_barrier", 50)
        )
        s["ai_risk"] = int(round(max(0, min(100, real_risk))))

    return data

# ==============================
# 4. User Interface & Visualization
# ==============================
if uploaded_file and gemini_key:
    if st.button("🚀 Analyze Skills & Forecast Trends", type="primary"):
        with st.spinner("Executing hybrid AI assessment and deterministic rubric calculation..."):
            try:
                cv_text = extract_text_from_pdf(uploaded_file)
                if not cv_text.strip():
                    st.error("Could not extract readable text from this PDF. Please ensure it is not a scanned image.")
                    st.stop()
                    
                result = analyze_skills_with_gemini(cv_text, gemini_key)

                # Overall Executive Summary
                st.info(f"💡 **Executive Profile Assessment:** {result.get('overall_summary')}")

                skills_data = result.get("skills", [])
                df = pd.DataFrame(skills_data)

                col1, col2 = st.columns([3, 2])

                # Left Column: Strategic Quadrant Scatter Plot
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
                            "growth_score": "Verified Growth Potential (0 - 100)", 
                            "ai_risk": "Verified AI Disruption Risk (0 - 100)",
                            "market_stage": "Market Phase"
                        },
                        title="Skill Positioning (Bottom-Right = High Growth & Defensible from AI)"
                    )
                    fig.update_traces(textposition='top center')
                    fig.update_layout(
                        xaxis=dict(range=[0, 105], dtick=20),
                        yaxis=dict(range=[0, 105], dtick=20),
                        legend=dict(orientation="h", y=-0.2)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Right Column: Explainable Rubric Breakdown
                with col2:
                    st.markdown("#### 🎯 Explainable Score Breakdown")
                    for s in skills_data:
                        with st.expander(f"{s['name']} — [{s['market_stage']}]"):
                            st.write(f"📈 **Growth Potential:** `{s['growth_score']}/100`")
                            gb = s.get("growth_breakdown", {})
                            st.caption("Formula: 40% Demand + 30% Ecosystem + 30% Salary")
                            st.progress(gb.get("demand_trajectory", 50) / 100, text=f"Hiring Demand (40%): {gb.get('demand_trajectory', 50)}/100")
                            st.progress(gb.get("ecosystem_alignment", 50) / 100, text=f"Ecosystem Alignment (30%): {gb.get('ecosystem_alignment', 50)}/100")
                            st.progress(gb.get("salary_premium", 50) / 100, text=f"Salary Premium (30%): {gb.get('salary_premium', 50)}/100")

                            st.write(f"🛡️ **AI Disruption Risk:** `{s['ai_risk']}/100`")
                            ab = s.get("ai_risk_breakdown", {})
                            st.caption("Formula: 40% Standardized + 40% Tools - 20% Judgment Barrier")
                            st.progress(ab.get("task_standardization", 50) / 100, text=f"Task Standardization (40%): {ab.get('task_standardization', 50)}/100")
                            st.progress(ab.get("ai_tool_maturity", 50) / 100, text=f"GenAI Tool Maturity (40%): {ab.get('ai_tool_maturity', 50)}/100")
                            st.progress(ab.get("human_judgment_barrier", 50) / 100, text=f"Human Judgment Barrier (-20% buffer): {ab.get('human_judgment_barrier', 50)}/100")

                            st.write(f"💬 **Rationale:** {s['reason']}")

                # Bottom Section: Recommended Upskilling Pathways
                st.divider()
                st.markdown("#### 🌟 Strategic Upskilling Pathways")
                st.caption("Recommended complementary capabilities to acquire next to maximize market defensibility:")
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
