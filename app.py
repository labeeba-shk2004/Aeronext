import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import dashboard components
from dashboard.overview_tab import render_overview_tab
from dashboard.recommendation_tab import render_recommendation_tab
from dashboard.pricing_tab import render_pricing_tab
from dashboard.sentiment_tab import render_sentiment_tab
from dashboard.comparison_tab import render_comparison_tab

# Import utilities
from simulation.data_generator import generate_all_data
from config.airport_profiles import AIRPORT_PROFILES

def main():
    # Configure page
    st.set_page_config(
        page_title="AeroNexus AI - Airport Revenue Optimization",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🛫 AeroNexus AI")
    st.markdown("AI-Driven Airport Revenue Optimization Platform")

    # Initialize data if not exists
    if not os.path.exists('data/delhi_passengers.csv'):
        with st.spinner('Initializing simulation data...'):
            generate_all_data()
    
    # Enhanced Sidebar with Logo
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <h2 style="color: white; margin: 0; font-family: 'Orbitron', monospace;">Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Airport selection
        st.subheader("✈️ Airport Selection")
        selected_airport = st.selectbox(
            "Choose Airport",
            options=list(AIRPORT_PROFILES.keys()),
            format_func=lambda x: f"{x} - {AIRPORT_PROFILES[x]['name']}"
        )
    
    # Store selected airport in session state
    if 'selected_airport' not in st.session_state:
        st.session_state.selected_airport = selected_airport
    
    st.session_state.selected_airport = selected_airport
    
    # Date range selector
    st.sidebar.subheader("📅 Analysis Period")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=30),
        max_value=datetime.now()
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now(),
        max_value=datetime.now()
    )
    
    # Store dates in session state
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    
    # Refresh data button
    if st.sidebar.button("🔄 Refresh Data", help="Generate new simulation data"):
        with st.spinner('Refreshing simulation data...'):
            generate_all_data()
        st.success("Data refreshed successfully!")
        st.rerun()
    
    # Main tabs with simplified labels
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "AI Recommendations", 
        "Dynamic Pricing",
        "Sentiment & NPS",
        "DEL vs JAI"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_recommendation_tab()
    
    with tab3:
        render_pricing_tab()
    
    with tab4:
        render_sentiment_tab()
    
    with tab5:
        render_comparison_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 AeroNext Phase 5 Pilot | Powered by AI & IoT | 
        Target: ₹312 → ₹390 revenue uplift | NPS +10 points</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
