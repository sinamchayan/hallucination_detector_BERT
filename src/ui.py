import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

from utils.config import UI_CONFIG
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Page config
st.set_page_config(
    page_title=UI_CONFIG['title'],
    page_icon=UI_CONFIG['page_icon'],
    layout=UI_CONFIG['layout']
)

API_URL = UI_CONFIG['api_url']

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


if 'history' not in st.session_state:
    st.session_state.history = []

def call_api(endpoint, data):
   
    try:
        response = requests.post(f"{API_URL}/{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        logger.error(f"API call failed: {str(e)}")
        return None

def get_metrics():
    #Get API Metrix
    try:
        response = requests.get(f"{API_URL}/metrics")
        return response.json()
    except:
        return None

def load_examples():
    
    try:
        response = requests.get(f"{API_URL}/examples")
        return response.json()['examples']
    except:
        return []

def main():
    
    st.title(UI_CONFIG['title'])
    st.markdown("""
    This system compares AI summaries against original text to identify:
    - ✅ **Correct** information (entailment)
    - ⚠️ **Unclear** claims (neutral - not enough info)
    - ❌ **Hallucinations** (contradictions - false information)
    """)
    
  
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
            st.info(f"Trying to connect to: {API_URL}")
        
        st.markdown("---")
        
        # Examples
        st.header("📖 Load Examples")
        examples = load_examples()
        
        for i, example in enumerate(examples):
            if st.button(f"📄 {example['name']}", key=f"example_{i}"):
                st.session_state.original = example['original_text']
                st.session_state.summary = example['summary_text']
                st.rerun()
        
        st.markdown("---")
        
        # Metrics
        st.header("📊 Statistics")
        metrics = get_metrics()
        if metrics and 'total_requests' in metrics:
            st.metric("Total Checks", metrics['total_requests'])
            st.metric("Hallucinations Found", metrics['total_hallucinations_detected'])
            st.metric("Avg Confidence", f"{metrics['average_confidence']:.2%}")
        elif metrics:
            st.info("Metrics not available")
        
        st.markdown("---")
        
        
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
        - MLflow tracking
        """)
    
    # Main content - tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Check", "📦 Batch Check", "📈 Analytics", "📜 History"])
    
    with tab1:
        st.header("Single Text Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Original Text")
            original = st.text_area(
                "Paste the source document",
                height=300,
                value=st.session_state.get('original', ''),
                placeholder="Enter the original text here...\n\nExample:\nApple announced the iPhone 15 in September 2023 with a USB-C port replacing the Lightning connector.",
                key='original_input'
            )
        
        with col2:
            st.subheader("✍️ AI-Generated Summary")
            summary = st.text_area(
                "Paste the summary to verify",
                height=300,
                value=st.session_state.get('summary', ''),
                placeholder="Enter the AI-generated summary here...\n\nExample:\nThe iPhone 15 features a USB-C port.",
                key='summary_input'
            )
        
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            check_button = st.button("🔍 Check for Hallucinations", type="primary", use_container_width=True)
        
        if check_button:
            if not original or not summary:
                st.error("⚠️ Please provide both original text and summary")
            else:
                with st.spinner("🤔 Analyzing..."):
                    # Call API
                    result = call_api('check', {
                        'original_text': original,
                        'summary_text': summary
                    })
                    
                    if result:
                        # Add to history
                        st.session_state.history.append({
                            'timestamp': result['timestamp'],
                            'original': original[:100] + '...',
                            'summary': summary[:100] + '...',
                            'result': result['result'],
                            'confidence': result['confidence']
                        })
                        
                      
                        st.markdown("---")
                        st.subheader("📊 Analysis Results")
                        
                        # Main result
                        if "HALLUCINATION" in result['result']:
                            st.error(f"### {result['result']}")
                            st.warning("⚠️ The summary contradicts the original text or contains false information!")
                        elif "CORRECT" in result['result']:
                            st.success(f"### {result['result']}")
                            st.info("✅ The summary is consistent with the original text.")
                        else:
                            st.warning(f"### {result['result']}")
                            st.info("⚪ Cannot determine with confidence. The summary may contain information not in the original.")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Overall Confidence", f"{result['confidence']:.1%}")
                        
                        with col2:
                            st.metric("✅ Correct", f"{result['all_scores']['correct']:.1%}")
                        
                        with col3:
                            st.metric("⚠️ Unclear", f"{result['all_scores']['unclear']:.1%}")
                        
                        with col4:
                            st.metric("❌ Hallucination", f"{result['all_scores']['hallucination']:.1%}")
                        
                        # Confidence breakdown chart
                        st.subheader("Confidence Breakdown")
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['Correct', 'Unclear', 'Hallucination'],
                                y=[
                                    result['all_scores']['correct'],
                                    result['all_scores']['unclear'],
                                    result['all_scores']['hallucination']
                                ],
                                marker_color=['green', 'orange', 'red']
                            )
                        ])
                        fig.update_layout(
                            yaxis_title="Confidence Score",
                            yaxis=dict(tickformat='.0%'),
                            showlegend=False,
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Processing info
                        with st.expander("⚙️ Processing Details"):
                            st.write(f"**Processing Time:** {result['processing_time_ms']:.2f}ms")
                            st.write(f"**Timestamp:** {result['timestamp']}")
                            st.write(f"**Prediction Index:** {result['prediction_idx']}")
    
    with tab2:
        st.header("Batch Processing")
        st.info("Check multiple summaries at once")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], help="CSV should have 'original_text' and 'summary_text' columns")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:")
            st.dataframe(df.head())
            
            if st.button("🚀 Process Batch", type="primary"):
                if 'original_text' in df.columns and 'summary_text' in df.columns:
                    with st.spinner(f"Processing {len(df)} pairs..."):
                        # Prepare data
                        text_pairs = [
                            {
                                'original_text': row['original_text'],
                                'summary_text': row['summary_text']
                            }
                            for _, row in df.iterrows()
                        ]
                        
                        # Call API
                        result = call_api('batch', {'text_pairs': text_pairs})
                        
                        if result:
                            st.success(f"✅ Processed {result['total_pairs']} pairs in {result['avg_processing_time_ms']:.2f}ms average")
                            
                            # Create results dataframe
                            results_df = pd.DataFrame([
                                {
                                    'Original': pair['original_text'][:50] + '...',
                                    'Summary': pair['summary_text'][:50] + '...',
                                    'Result': r['result'],
                                    'Confidence': f"{r['confidence']:.2%}"
                                }
                                for pair, r in zip(text_pairs, result['results'])
                            ])
                            
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                csv,
                                "batch_results.csv",
                                "text/csv"
                            )
                else:
                    st.error("CSV must have 'original_text' and 'summary_text' columns")
        else:
          
            st.subheader("Or enter manually:")
            num_pairs = st.number_input("Number of pairs", min_value=1, max_value=10, value=2)
            
            pairs = []
            for i in range(num_pairs):
                st.markdown(f"**Pair {i+1}**")
                col1, col2 = st.columns(2)
                with col1:
                    orig = st.text_input(f"Original {i+1}", key=f"batch_orig_{i}")
                with col2:
                    summ = st.text_input(f"Summary {i+1}", key=f"batch_summ_{i}")
                
                if orig and summ:
                    pairs.append({'original_text': orig, 'summary_text': summ})
            
            if st.button("Check All", type="primary") and len(pairs) == num_pairs:
                with st.spinner("Processing..."):
                    result = call_api('batch', {'text_pairs': pairs})
                    if result:
                        for i, r in enumerate(result['results']):
                            st.write(f"**Pair {i+1}:** {r['result']} ({r['confidence']:.2%})")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        metrics = get_metrics()
        if metrics and isinstance(metrics, dict) and metrics.get('total_requests', 0) > 0:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Checks",
                    metrics['total_requests'],
                    help="Total number of hallucination checks performed"
                )
            
            with col2:
                st.metric(
                    "Hallucinations Found",
                    metrics['total_hallucinations_detected'],
                    help="Number of hallucinations detected"
                )
            
            with col3:
                st.metric(
                    "Detection Rate",
                    f"{metrics['hallucination_rate']:.1%}",
                    help="Percentage of summaries with hallucinations"
                )
            
            with col4:
                st.metric(
                    "Avg Confidence",
                    f"{metrics['average_confidence']:.1%}",
                    help="Average confidence across all predictions"
                )
            
            # Charts
            if st.session_state.history:
                st.subheader("Recent Activity")
                
                # Results distribution
                history_df = pd.DataFrame(st.session_state.history)
                result_counts = history_df['result'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        values=result_counts.values,
                        names=result_counts.index,
                        title="Results Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        history_df,
                        x='confidence',
                        nbins=20,
                        title="Confidence Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet. Perform some checks to see analytics!")
    
    with tab4:
        st.header("Check History")
        
        if st.session_state.history:
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
            
            # Display history
            for i, item in enumerate(reversed(st.session_state.history)):
                with st.expander(f"{item['timestamp']} - {item['result']}"):
                    st.write(f"**Original:** {item['original']}")
                    st.write(f"**Summary:** {item['summary']}")
                    st.write(f"**Result:** {item['result']}")
                    st.write(f"**Confidence:** {item['confidence']:.2%}")
        else:
            st.info("No history yet. Start checking summaries!")

if __name__ == '__main__':
    main()