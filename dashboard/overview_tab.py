"""
Overview tab for the AeroNexus AI dashboard.
Displays airport profiles, baseline metrics, and key performance indicators.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.airport_profiles import AIRPORT_PROFILES, KPI_TARGETS
from utils.kpi_calculator import KPICalculator
from utils.plot_utils import create_gauge_chart, create_metric_comparison

def render_overview_tab():
    """Render the overview tab with airport profiles and KPIs"""
    
    # Get selected airport from session state
    selected_airport = st.session_state.get('selected_airport', 'DEL')
    airport_profile = AIRPORT_PROFILES[selected_airport]
    
    # Header section
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #1f77b4, #ff7f0e); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>{airport_profile['name']} ({selected_airport})</h2>
        <p style='color: white; margin: 10px 0 0 0; font-size: 1.1em;'>
            {airport_profile['passenger_volume']:,} passengers annually | 
            {airport_profile['tech_readiness']} technology readiness
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Baseline Revenue/Pax", 
            f"₹{airport_profile['baseline_revenue_per_pax']}", 
            f"Target: ₹{airport_profile['target_revenue_per_pax']}"
        )
    
    with col2:
        st.metric(
            "Current NPS", 
            f"{airport_profile['baseline_nps']}", 
            f"Target: {airport_profile['target_nps']}"
        )
    
    with col3:
        st.metric(
            "Avg Dwell Time", 
            f"{airport_profile['baseline_dwell_time']} min", 
            f"±{airport_profile['dwell_time_variance']} min variance"
        )
    
    with col4:
        retail_density = airport_profile['retail_density']
        st.metric(
            "Retail Density", 
            f"{retail_density}/1000 pax", 
            "shops per 1000 passengers"
        )
    
    st.markdown("---")
    
    # Two main columns for detailed view
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("Pilot Objectives & Targets")
        
        # Revenue target progress
        current_revenue = airport_profile['baseline_revenue_per_pax']
        target_revenue = airport_profile['target_revenue_per_pax']
        
        # Simulate some progress (in real implementation, this would come from actual data)
        kpi_calc = KPICalculator()
        
        # Create progress visualization
        progress_data = {
            'Revenue per Passenger': {
                'current': current_revenue,
                'target': target_revenue,
                'unit': '₹'
            },
            'Net Promoter Score': {
                'current': airport_profile['baseline_nps'],
                'target': airport_profile['target_nps'],
                'unit': ''
            },
            'Dwell Time Variance': {
                'current': airport_profile['dwell_time_variance'],
                'target': airport_profile['target_dwell_variance'],
                'unit': '±min',
                'reverse': True  # Lower is better
            }
        }
        
        for metric, data in progress_data.items():
            current = data['current']
            target = data['target']
            unit = data['unit']
            reverse = data.get('reverse', False)
            
            if reverse:
                progress = max(0, (current - target) / current * 100)
                progress = 100 - progress
            else:
                progress = min(100, (current / target) * 100)
            
            # Progress bar
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = current,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"{metric}"},
                delta = {'reference': target, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, target * 1.2 if not reverse else current * 1.2]},
                    'bar': {'color': "lightblue"},
                    'steps': [
                        {'range': [0, target * 0.8 if not reverse else current * 0.8], 'color': "lightgray"},
                        {'range': [target * 0.8 if not reverse else current * 0.8, target if not reverse else current], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': target if not reverse else current * 0.7
                    }
                }
            ))
            
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    with right_col:
        st.subheader("Passenger Segment Distribution")
        
        # Passenger segment pie chart
        segments = {
            'Business': airport_profile['business_segment'],
            'Leisure': airport_profile['leisure_segment']
        }
        
        fig_segments = px.pie(
            values=list(segments.values()),
            names=list(segments.keys()),
            title=f"Passenger Segments - {selected_airport}",
            color_discrete_map={
                'Business': '#1f77b4',
                'Leisure': '#ff7f0e'
            }
        )
        fig_segments.update_traces(textposition='inside', textinfo='percent+label')
        fig_segments.update_layout(height=300)
        st.plotly_chart(fig_segments, use_container_width=True)
        
        st.subheader("Retail Category Mix")
        
        # Retail categories bar chart
        categories = airport_profile['retail_categories']
        
        fig_retail = px.bar(
            x=list(categories.keys()),
            y=[v * 100 for v in categories.values()],
            title="Retail Category Distribution (%)",
            labels={'x': 'Category', 'y': 'Percentage (%)'},
            color=list(categories.values()),
            color_continuous_scale='viridis'
        )
        fig_retail.update_layout(
            height=300,
            xaxis_tickangle=-45,
            showlegend=False
        )
        st.plotly_chart(fig_retail, use_container_width=True)
    
    st.markdown("---")
    
    # Operational insights section
    st.subheader("Operational Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.markdown("""
        **Revenue Opportunity**
        - Current spend: ₹{:,}/passenger
        - Target uplift: {:,}%
        - Potential gain: ₹{:,.0f}/passenger
        """.format(
            current_revenue,
            KPI_TARGETS['revenue_uplift_percent'],
            target_revenue - current_revenue
        ))
    
    with insight_col2:
        st.markdown("""
        **Experience Improvement**
        - Current NPS: {} points
        - Target improvement: +{} points
        - Satisfaction focus: Digital engagement
        """.format(
            airport_profile['baseline_nps'],
            KPI_TARGETS['nps_improvement']
        ))
    
    with insight_col3:
        st.markdown("""
        **Efficiency Gains**
        - Dwell time variance: ±{} min
        - Target reduction: {}%
        - Technology: IoT sensors + AI
        """.format(
            airport_profile['dwell_time_variance'],
            KPI_TARGETS['dwell_time_variance_reduction']
        ))
    
    # Commercial zones section
    st.subheader("Commercial Zones")
    
    zones_data = []
    for i, zone in enumerate(airport_profile['commercial_zones']):
        # Simulate utilization data (in real implementation, this would come from IoT sensors)
        utilization = np.random.uniform(60, 95)
        foot_traffic = np.random.randint(100, 500)
        
        zones_data.append({
            'Zone': zone,
            'Utilization (%)': round(utilization, 1),
            'Foot Traffic/Hour': foot_traffic,
            'Primary Category': list(airport_profile['retail_categories'].keys())[i % len(airport_profile['retail_categories'])]
        })
    
    zones_df = pd.DataFrame(zones_data)
    
    # Display zones in expandable format
    for _, zone in zones_df.iterrows():
        with st.expander(f"{zone['Zone']}"):
            zone_col1, zone_col2, zone_col3 = st.columns(3)
            with zone_col1:
                st.metric("Space Utilization", f"{zone['Utilization (%)']}%")
            with zone_col2:
                st.metric("Foot Traffic", f"{zone['Foot Traffic/Hour']}/hr")
            with zone_col3:
                st.metric("Primary Category", zone['Primary Category'])
    
    # Peak hours visualization
    st.subheader("Peak Hours Analysis")
    
    peak_hours = airport_profile['peak_hours']
    hours_data = []
    
    for hour in range(24):
        traffic_level = 'Low'
        multiplier = 0.3
        
        for period, (start, end) in peak_hours.items():
            if start <= hour <= end:
                traffic_level = 'High'
                multiplier = 1.0
                break
        
        if traffic_level == 'Low' and 10 <= hour <= 18:
            traffic_level = 'Medium'
            multiplier = 0.6
        
        hours_data.append({
            'Hour': f"{hour:02d}:00",
            'Traffic Level': traffic_level,
            'Relative Traffic': multiplier * 100,
            'Hour_Numeric': hour
        })
    
    hours_df = pd.DataFrame(hours_data)
    
    fig_hours = px.bar(
        hours_df,
        x='Hour_Numeric',
        y='Relative Traffic',
        color='Traffic Level',
        title="Hourly Traffic Patterns",
        labels={'Hour_Numeric': 'Hour of Day', 'Relative Traffic': 'Traffic Level (%)'},
        color_discrete_map={
            'Low': '#ff7f7f',
            'Medium': '#ffb347',
            'High': '#90ee90'
        }
    )
    
    fig_hours.update_layout(
        height=400,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]
        )
    )
    
    st.plotly_chart(fig_hours, use_container_width=True)
    
    # Success metrics summary
    st.markdown("---")
    st.subheader("Success Metrics Summary")
    
    metrics_data = [
        {"Metric": "Revenue Uplift", "Baseline": f"₹{current_revenue}", "Target": f"₹{target_revenue}", "Impact": f"+{((target_revenue/current_revenue - 1) * 100):.1f}%"},
        {"Metric": "NPS Improvement", "Baseline": f"{airport_profile['baseline_nps']}", "Target": f"{airport_profile['target_nps']}", "Impact": f"+{airport_profile['target_nps'] - airport_profile['baseline_nps']} pts"},
        {"Metric": "Dwell Time Variance", "Baseline": f"±{airport_profile['dwell_time_variance']} min", "Target": f"±{airport_profile['target_dwell_variance']} min", "Impact": f"-{KPI_TARGETS['dwell_time_variance_reduction']}%"},
        {"Metric": "Retail Conversion", "Baseline": "15%", "Target": "25%", "Impact": f"+{KPI_TARGETS['retail_conversion_uplift']}%"},
    ]
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Call to action
    st.info("""
    💡 **Next Steps**: Explore the AI Recommendations tab to see personalized offers in action, 
    or check Dynamic Pricing to understand revenue optimization strategies.
    """)
