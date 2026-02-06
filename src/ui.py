"""Streamlit UI for Hallucination Detection"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

from utils.config import UI_CONFIG

# Page config
st.set_page_config(
    page_title=UI_CONFIG['title'],
    page_icon=UI_CONFIG['page_icon'],
    layout=UI_CONFIG['layout']
)

API_URL = UI_CONFIG['api_url']

# Session state
if 'history' not in st.session_state:
    st.session_state.history = []

def call_api(endpoint, data):
    """Call API endpoint"""
    try:
        response = requests.post(f"{API_URL}/{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def load_examples():
    """Load example texts from API"""
    try:
        response = requests.get(f"{API_URL}/examples")
        return response.json()['examples']
    except:
        return []

# Main UI
st.title(UI_CONFIG['title'])
st.markdown("**Detect hallucinations in AI-generated summaries using Natural Language Inference**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API health check
    try:
        health = requests.get(f"{API_URL}/health").json()
        if health['status'] == 'healthy':
            st.success("✅ API Connected")
        else:
            st.error("❌ API Unavailable")
    except:
        st.error("❌ Cannot connect to API")
        st.info(f"Make sure API is running at: {API_URL}")
    
    st.markdown("---")
    
    # Load examples
    st.header("📖 Load Examples")
    examples = load_examples()
    
    for i, example in enumerate(examples):
        if st.button(f"📄 {example['name']}", key=f"example_{i}"):
            st.session_state.original = example['original_text']
            st.session_state.summary = example['summary_text']
            st.rerun()
    
    st.markdown("---")
    
    # About
    st.header("ℹ️ About")
    st.info("""
    **How it works:**
    1. Model trained on 50k examples
    2. Uses BERT for NLI
    3. Compares semantic meaning
    4. Detects contradictions
    
    **Tech Stack:**
    - TensorFlow + BERT
    - FastAPI backend
    - Streamlit frontend
    """)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Original Text")
    original = st.text_area(
        "Paste the source document",
        height=300,
        value=st.session_state.get('original', ''),
        placeholder="Enter original text here..."
    )

with col2:
    st.subheader("✍️ AI-Generated Summary")
    summary = st.text_area(
        "Paste the summary to verify",
        height=300,
        value=st.session_state.get('summary', ''),
        placeholder="Enter AI summary here..."
    )

# Check button
if st.button("🔍 Check for Hallucinations", type="primary"):
    if not original or not summary:
        st.error("⚠️ Please provide both original text and summary")
    else:
        with st.spinner("🤔 Analyzing..."):
            result = call_api('check', {
                'original_text': original,
                'summary_text': summary
            })
            
            if result:
                # Add to history
                st.session_state.history.append(result)
                
                # Display result
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                if "HALLUCINATION" in result['result']:
                    st.error(f"### {result['result']}")
                    st.warning("⚠️ The summary contradicts the original text!")
                elif "CORRECT" in result['result']:
                    st.success(f"### {result['result']}")
                    st.info("✅ The summary is consistent with the original.")
                else:
                    st.warning(f"### {result['result']}")
                    st.info("⚪ Cannot determine with confidence.")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("✅ Correct", f"{result['all_scores']['correct']:.1%}")
                with col2:
                    st.metric("⚠️ Unclear", f"{result['all_scores']['unclear']:.1%}")
                with col3:
                    st.metric("❌ Hallucination", f"{result['all_scores']['hallucination']:.1%}")
                
                # Chart
                st.subheader("Confidence Breakdown")
                fig = px.bar(
                    x=['Correct', 'Unclear', 'Hallucination'],
                    y=[
                        result['all_scores']['correct'],
                        result['all_scores']['unclear'],
                        result['all_scores']['hallucination']
                    ],
                    color=['Correct', 'Unclear', 'Hallucination'],
                    color_discrete_map={
                        'Correct': 'green',
                        'Unclear': 'orange',
                        'Hallucination': 'red'
                    }
                )
                fig.update_layout(showlegend=False, yaxis_title="Score", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Details
                with st.expander("⚙️ Processing Details"):
                    st.write(f"**Processing Time:** {result['processing_time_ms']:.2f}ms")
                    st.write(f"**Timestamp:** {result['timestamp']}")

if __name__ == '__main__':
    pass
