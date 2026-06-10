"""
Sentiment Analysis & NPS tab for the AeroNexus AI dashboard.
Processes passenger feedback to derive sentiment and calculate NPS scores.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_nlp import SentimentNLPAnalyzer
from config.airport_profiles import AIRPORT_PROFILES, KPI_TARGETS
from utils.plot_utils import create_nps_gauge, create_sentiment_trend_chart

def render_sentiment_tab():
    """Render the sentiment analysis and NPS tab"""
    
    st.header("Sentiment Analysis & NPS Tracking")
    st.markdown("""
    Real-time passenger feedback analysis using NLP to track sentiment trends 
    and calculate Net Promoter Scores for continuous experience improvement.
    """)
    
    # Get selected airport
    selected_airport = st.session_state.get('selected_airport', 'DEL')
    airport_profile = AIRPORT_PROFILES[selected_airport]
    
    # Initialize sentiment analyzer
    @st.cache_resource
    def load_sentiment_analyzer():
        analyzer = SentimentNLPAnalyzer()
        return analyzer
    
    sentiment_analyzer = load_sentiment_analyzer()
    
    # Load feedback data
    @st.cache_data
    def load_feedback_data():
        try:
            feedback_df = pd.read_csv('data/feedback.csv')
            # Filter by selected airport
            airport_feedback = feedback_df[feedback_df['airport_code'] == selected_airport]
            return airport_feedback
        except FileNotFoundError:
            return pd.DataFrame()
    
    feedback_df = load_feedback_data()
    
    # Current NPS Overview
    st.subheader("Current NPS Overview")
    
    if not feedback_df.empty:
        # Calculate current NPS
        feedback_list = feedback_df['feedback_text'].tolist()
        nps_results = sentiment_analyzer.calculate_nps_from_feedback(feedback_list)
        
        current_nps = nps_results['nps']
        baseline_nps = airport_profile['baseline_nps']
        target_nps = airport_profile['target_nps']
        
        # NPS metrics display
        nps_col1, nps_col2, nps_col3, nps_col4 = st.columns(4)
        
        with nps_col1:
            st.metric(
                "Current NPS",
                f"{current_nps:.1f}",
                f"{current_nps - baseline_nps:+.1f} vs baseline"
            )
        
        with nps_col2:
            st.metric(
                "Target Progress",
                f"{((current_nps - baseline_nps) / (target_nps - baseline_nps) * 100):.1f}%",
                f"Goal: {target_nps}"
            )
        
        with nps_col3:
            st.metric(
                "Promoters",
                f"{nps_results['promoters']}",
                f"{nps_results['promoter_percentage']:.1f}%"
            )
        
        with nps_col4:
            st.metric(
                "Detractors", 
                f"{nps_results['detractors']}",
                f"{nps_results['detractor_percentage']:.1f}%"
            )
        
        # NPS Gauge Chart
        nps_gauge = create_nps_gauge(current_nps, target_nps)
        st.plotly_chart(nps_gauge, use_container_width=True)
        
        # NPS Distribution
        st.subheader("NPS Distribution")
        
        distribution_data = {
            'Category': ['Promoters (9-10)', 'Passives (7-8)', 'Detractors (0-6)'],
            'Count': [nps_results['promoters'], nps_results['passives'], nps_results['detractors']],
            'Percentage': [nps_results['promoter_percentage'], 
                          round((nps_results['passives'] / nps_results['total_responses']) * 100, 1),
                          nps_results['detractor_percentage']]
        }
        
        fig_distribution = px.bar(
            x=distribution_data['Category'],
            y=distribution_data['Count'],
            color=distribution_data['Category'],
            title="NPS Score Distribution",
            color_discrete_map={
                'Promoters (9-10)': '#28a745',
                'Passives (7-8)': '#ffc107', 
                'Detractors (0-6)': '#dc3545'
            },
            text=distribution_data['Percentage']
        )
        fig_distribution.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_distribution.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_distribution, use_container_width=True)
        
    else:
        st.warning("No feedback data available for the selected airport. Please ensure feedback data is loaded.")
        # Show baseline metrics instead
        nps_col1, nps_col2 = st.columns(2)
        with nps_col1:
            st.metric("Baseline NPS", f"{airport_profile['baseline_nps']}")
        with nps_col2:
            st.metric("Target NPS", f"{airport_profile['target_nps']}")
    
    st.markdown("---")
    
    # Real-time Feedback Analysis
    st.subheader("Real-time Feedback Analysis")
    
    # Text input for new feedback
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_feedback = st.text_area(
            "Enter passenger feedback for analysis:",
            placeholder="The airport shopping experience was excellent! Staff was very helpful and the digital signage made navigation easy.",
            height=120
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        analyze_button = st.button("Analyze Sentiment", type="primary")
        
        simulate_feedback = st.button("Generate Sample", help="Generate sample feedback for testing")
    
    if simulate_feedback:
        from simulation.feedback_simulator import FeedbackSimulator
        feedback_sim = FeedbackSimulator()
        sample_feedback = feedback_sim.generate_feedback(selected_airport, 'mixed')
        user_feedback = sample_feedback['feedback_text']
        st.rerun()
    
    if analyze_button and user_feedback.strip():
        with st.spinner("Analyzing feedback sentiment..."):
            
            # Analyze sentiment
            sentiment_result = sentiment_analyzer.analyze_sentiment(user_feedback)
            nps_result = sentiment_analyzer.sentiment_to_nps(
                sentiment_result['polarity'], 
                sentiment_result['confidence']
            )
            aspect_analysis = sentiment_analyzer.analyze_aspects(user_feedback)
            
            # Display results
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                sentiment_color = "green" if sentiment_result['sentiment_label'] == 'positive' else "red" if sentiment_result['sentiment_label'] == 'negative' else "orange"
                st.markdown(f"**Sentiment:** :{sentiment_color}[{sentiment_result['sentiment_label'].title()}]")
                st.metric("Polarity Score", f"{sentiment_result['polarity']:.3f}", "(-1 to +1 scale)")
            
            with result_col2:
                nps_color = "green" if nps_result['nps_category'] == 'promoter' else "red" if nps_result['nps_category'] == 'detractor' else "orange"
                st.markdown(f"**NPS Category:** :{nps_color}[{nps_result['nps_category'].title()}]")
                st.metric("Predicted NPS", f"{nps_result['nps_score']:.1f}", "0-10 scale")
            
            with result_col3:
                st.metric("Confidence", f"{sentiment_result['confidence']:.1%}", "Analysis reliability")
                st.metric("Subjectivity", f"{sentiment_result['subjectivity']:.3f}", "0=objective, 1=subjective")
            
            # Aspect Analysis
            if aspect_analysis:
                st.subheader("Aspect-wise Analysis")
                
                aspect_data = []
                for aspect, data in aspect_analysis.items():
                    aspect_data.append({
                        'Aspect': aspect.replace('_', ' ').title(),
                        'Sentiment': data['sentiment'],
                        'Mentions': data['mentions'],
                        'Confidence': data['confidence']
                    })
                
                aspect_df = pd.DataFrame(aspect_data)
                
                # Aspect sentiment chart
                fig_aspects = px.bar(
                    aspect_df,
                    x='Aspect',
                    y='Sentiment',
                    color='Sentiment',
                    title="Sentiment by Aspect",
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0
                )
                fig_aspects.update_layout(height=350, xaxis_tickangle=-45)
                st.plotly_chart(fig_aspects, use_container_width=True)
                
                # Detailed aspect table
                st.dataframe(aspect_df, use_container_width=True)
    
    # Sentiment Trends Analysis
    if not feedback_df.empty:
        st.markdown("---")
        st.subheader("Sentiment Trends Analysis")
        
        # Ensure timestamp column exists and convert
        if 'timestamp' not in feedback_df.columns:
            feedback_df['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(days=30),
                periods=len(feedback_df),
                freq='H'
            )
        else:
            feedback_df['timestamp'] = pd.to_datetime(feedback_df['timestamp'])
        
        # Analyze trends
        trends = sentiment_analyzer.analyze_feedback_trends(feedback_df)
        
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            if 'daily_sentiment' in trends and trends['daily_sentiment']:
                daily_data = pd.DataFrame([
                    {'Date': date, 'Sentiment': sentiment} 
                    for date, sentiment in trends['daily_sentiment'].items()
                ])
                daily_data['Date'] = pd.to_datetime(daily_data['Date'])
                
                fig_daily = px.line(
                    daily_data,
                    x='Date',
                    y='Sentiment',
                    title="Daily Sentiment Trend",
                    markers=True
                )
                fig_daily.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
                fig_daily.update_layout(height=350)
                st.plotly_chart(fig_daily, use_container_width=True)
        
        with trend_col2:
            if 'hourly_pattern' in trends and trends['hourly_pattern']:
                hourly_data = pd.DataFrame([
                    {'Hour': hour, 'Sentiment': sentiment}
                    for hour, sentiment in trends['hourly_pattern'].items()
                ])
                
                fig_hourly = px.bar(
                    hourly_data,
                    x='Hour',
                    y='Sentiment',
                    title="Hourly Sentiment Pattern",
                    color='Sentiment',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0
                )
                fig_hourly.update_layout(height=350)
                st.plotly_chart(fig_hourly, use_container_width=True)
        
        # Common themes
        if 'common_themes' in trends and trends['common_themes']:
            st.subheader("Common Themes")
            
            themes_data = pd.DataFrame([
                {'Theme': theme, 'Frequency': freq}
                for theme, freq in trends['common_themes'].items()
            ])
            
            fig_themes = px.bar(
                themes_data,
                x='Frequency',
                y='Theme',
                orientation='h',
                title="Most Common Feedback Themes",
                color='Frequency',
                color_continuous_scale='viridis'
            )
            fig_themes.update_layout(height=400)
            st.plotly_chart(fig_themes, use_container_width=True)
    
    # Actionable Insights
    st.markdown("---")
    st.subheader("Actionable Insights & Recommendations")
    
    if not feedback_df.empty:
        # Generate insights
        feedback_list = [{'feedback_text': text} for text in feedback_df['feedback_text'].tolist()]
        insights = sentiment_analyzer.generate_insights(feedback_list, selected_airport)
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("**Summary Statistics**")
            if insights['summary']:
                st.write(f"• Average Sentiment: {insights['summary']['average_sentiment']:.3f}")
                st.write(f"• Total Feedback Analyzed: {insights['summary']['total_feedback']:,}")
                st.write(f"• Sentiment Variability: {insights['summary']['sentiment_std']:.3f}")
            
            if insights['sentiment_distribution']:
                st.markdown("**Sentiment Distribution**")
                total = sum(insights['sentiment_distribution'].values())
                for sentiment, count in insights['sentiment_distribution'].items():
                    percentage = (count / total) * 100 if total > 0 else 0
                    st.write(f"• {sentiment.title()}: {count} ({percentage:.1f}%)")
        
        with insight_col2:
            st.markdown("**Priority Areas for Improvement**")
            if insights['priority_areas']:
                for area in insights['priority_areas'][:5]:
                    st.write(f"• **{area['aspect'].replace('_', ' ').title()}**: {area['score']:.3f} ({area['mentions']} mentions)")
            else:
                st.write("No critical areas identified - good performance!")
            
            st.markdown("**Recommended Actions**")
            if insights['recommendations']:
                for rec in insights['recommendations']:
                    st.write(f"• {rec}")
            else:
                st.write("Continue current excellence in service delivery!")
    
    else:
        # Show general insights for the airport
        st.info(f"""
        **Insights for {airport_profile['name']}:**
        
        • **Baseline NPS**: {airport_profile['baseline_nps']} (Industry benchmark)
        • **Target Improvement**: +{KPI_TARGETS['nps_improvement']} points 
        • **Key Focus Areas**: Digital experience, retail personalization, service quality
        • **Success Metrics**: Track sentiment trends, aspect-wise feedback, real-time NPS
        """)
    
    # NPS Prediction
    st.markdown("---")
    st.subheader("NPS Trend Prediction")
    
    if not feedback_df.empty and 'daily_sentiment' in trends:
        # Predict NPS trend
        current_nps_for_prediction = nps_results.get('nps', airport_profile['baseline_nps'])
        predictions = sentiment_analyzer.predict_nps_trend(
            current_nps=current_nps_for_prediction,
            sentiment_trend=trends['daily_sentiment'],
            days_ahead=30
        )
        
        if predictions:
            pred_data = pd.DataFrame(predictions)
            pred_data['Date'] = [datetime.now() + timedelta(days=day) for day in pred_data['day']]
            
            fig_prediction = px.line(
                pred_data,
                x='Date',
                y='predicted_nps',
                title="30-Day NPS Prediction",
                markers=True
            )
            
            # Add current NPS line
            fig_prediction.add_hline(
                y=current_nps_for_prediction,
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Current NPS: {current_nps_for_prediction:.1f}"
            )
            
            # Add target NPS line
            fig_prediction.add_hline(
                y=target_nps,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Target NPS: {target_nps}"
            )
            
            fig_prediction.update_layout(height=400)
            st.plotly_chart(fig_prediction, use_container_width=True)
            
            # Prediction confidence
            avg_confidence = pred_data['confidence'].mean()
            st.info(f"**Prediction Confidence**: {avg_confidence:.1%} - Based on recent sentiment trends and historical patterns")
    
    else:
        st.info("NPS prediction requires historical feedback data. Continue collecting feedback for trend analysis.")
    
    # Feedback Collection Tools
    st.markdown("---")
    st.subheader("Feedback Collection & Management")
    
    collection_col1, collection_col2 = st.columns(2)
    
    with collection_col1:
        st.markdown("**Collection Channels**")
        channels = ['Mobile App', 'Website Survey', 'Email Follow-up', 'SMS', 'QR Code', 'Kiosk']
        
        for channel in channels:
            # Simulate channel performance
            responses = np.random.randint(50, 500)
            response_rate = np.random.uniform(15, 45)
            st.write(f"• **{channel}**: {responses} responses ({response_rate:.1f}% rate)")
    
    with collection_col2:
        st.markdown("**Response Quality Metrics**")
        quality_metrics = {
            'Average Response Length': f"{np.random.randint(25, 80)} words",
            'Sentiment Clarity': f"{np.random.uniform(75, 95):.1f}%",
            'Actionable Feedback': f"{np.random.uniform(60, 85):.1f}%",
            'Multi-aspect Coverage': f"{np.random.uniform(40, 70):.1f}%"
        }
        
        for metric, value in quality_metrics.items():
            st.write(f"• **{metric}**: {value}")
    
    # Export functionality
    if st.button("Export Sentiment Analysis Report"):
        if not feedback_df.empty:
            # Create comprehensive report
            report_data = []
            
            for _, row in feedback_df.iterrows():
                sentiment_result = sentiment_analyzer.analyze_sentiment(row['feedback_text'])
                nps_result = sentiment_analyzer.sentiment_to_nps(
                    sentiment_result['polarity'],
                    sentiment_result['confidence']
                )
                
                report_data.append({
                    'Feedback_ID': row.get('feedback_id', f"FB_{len(report_data)+1:03d}"),
                    'Passenger_ID': row.get('passenger_id', 'Unknown'),
                    'Timestamp': row.get('timestamp', datetime.now()),
                    'Feedback_Text': row['feedback_text'],
                    'Sentiment_Label': sentiment_result['sentiment_label'],
                    'Sentiment_Score': sentiment_result['polarity'],
                    'Confidence': sentiment_result['confidence'],
                    'NPS_Score': nps_result['nps_score'],
                    'NPS_Category': nps_result['nps_category'],
                    'Airport_Code': selected_airport
                })
            
            report_df = pd.DataFrame(report_data)
            csv = report_df.to_csv(index=False)
            
            st.download_button(
                label="Download Detailed Report",
                data=csv,
                file_name=f"sentiment_analysis_{selected_airport}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No feedback data available to export.")
