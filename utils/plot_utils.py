"""
Plotting utilities for the AeroNexus AI dashboard.
Centralized functions for creating consistent, reusable visualizations.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_gauge_chart(value, title, min_val=0, max_val=100, target=None, 
                      color_ranges=None, suffix="", height=300):
    """Create a gauge chart for KPI visualization"""
    
    if color_ranges is None:
        color_ranges = [
            {"range": [min_val, max_val * 0.5], "color": "lightgray"},
            {"range": [max_val * 0.5, max_val * 0.8], "color": "yellow"},
            {"range": [max_val * 0.8, max_val], "color": "green"}
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': target} if target else None,
        gauge={
            'axis': {'range': [None, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [{'range': [r["range"][0], r["range"][1]], 'color': r["color"]} 
                     for r in color_ranges],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': target if target else max_val * 0.9
            }
        },
        number={'suffix': suffix}
    ))
    
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_nps_gauge(current_nps, target_nps, height=400):
    """Create an NPS-specific gauge chart"""
    
    color_ranges = [
        {"range": [-100, 0], "color": "#ff4444"},      # Detractor zone
        {"range": [0, 50], "color": "#ffaa00"},        # Passive zone  
        {"range": [50, 100], "color": "#00aa00"}       # Promoter zone
    ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_nps,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Net Promoter Score (NPS)"},
        delta={'reference': target_nps, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-100, 0], 'color': '#ffcccc'},
                {'range': [0, 50], 'color': '#ffffcc'},
                {'range': [50, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': target_nps
            }
        }
    ))
    
    # Add NPS zone annotations
    fig.add_annotation(x=0.2, y=0.2, text="Detractors<br>(-100 to 0)", 
                      showarrow=False, font=dict(size=10, color="red"))
    fig.add_annotation(x=0.5, y=0.1, text="Passives<br>(1 to 50)", 
                      showarrow=False, font=dict(size=10, color="orange"))
    fig.add_annotation(x=0.8, y=0.2, text="Promoters<br>(51 to 100)", 
                      showarrow=False, font=dict(size=10, color="green"))
    
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_metric_comparison(metrics_data, title="Metric Comparison", 
                           x_col="Metric", y_cols=None, height=400):
    """Create a bar chart comparing metrics across different categories"""
    
    if y_cols is None:
        y_cols = [col for col in metrics_data.columns if col != x_col]
    
    fig = px.bar(
        metrics_data,
        x=x_col,
        y=y_cols,
        title=title,
        barmode='group'
    )
    
    fig.update_layout(
        height=height,
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_recommendation_chart(recommendations, height=400):
    """Create visualization for recommendation results"""
    
    if not recommendations:
        fig = go.Figure()
        fig.add_annotation(
            text="No recommendations available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(height=height)
        return fig
    
    # Extract data for visualization
    product_ids = [rec['product_id'] for rec in recommendations]
    ratings = [rec['predicted_rating'] for rec in recommendations]
    confidence = [rec['confidence'] for rec in recommendations]
    
    # Create subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add rating bars
    fig.add_trace(
        go.Bar(name="Affinity Score", x=product_ids, y=ratings, 
              marker_color='lightblue', opacity=0.8),
        secondary_y=False,
    )
    
    # Add confidence line
    fig.add_trace(
        go.Scatter(name="Confidence", x=product_ids, y=confidence, 
                  mode='lines+markers', marker_color='red', line=dict(width=3)),
        secondary_y=True,
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Affinity Score (1-5)", secondary_y=False)
    fig.update_yaxes(title_text="Confidence (0-1)", secondary_y=True)
    
    fig.update_layout(
        title="Product Recommendations - Affinity vs Confidence",
        xaxis_title="Product ID",
        height=height
    )
    
    return fig

def create_pricing_chart(pricing_data, title="Dynamic Pricing Analysis", height=500):
    """Create pricing analysis visualization with dual y-axis"""
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=pricing_data['hour'],
            y=pricing_data['adjusted_price'],
            mode='lines+markers',
            name='Dynamic Price',
            line=dict(color='blue', width=3)
        ),
        secondary_y=False,
    )
    
    # Add baseline price
    if 'base_price' in pricing_data.columns:
        fig.add_trace(
            go.Scatter(
                x=pricing_data['hour'],
                y=pricing_data['base_price'],
                mode='lines',
                name='Base Price',
                line=dict(color='gray', dash='dash', width=2)
            ),
            secondary_y=False,
        )
    
    # Add sales volume bars
    if 'actual_sales' in pricing_data.columns:
        fig.add_trace(
            go.Bar(
                x=pricing_data['hour'],
                y=pricing_data['actual_sales'],
                name='Sales Volume',
                opacity=0.3,
                marker_color='green'
            ),
            secondary_y=True,
        )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Price (₹)", secondary_y=False)
    fig.update_yaxes(title_text="Sales Volume", secondary_y=True)
    
    fig.update_layout(
        title=title,
        xaxis_title="Hour of Day",
        height=height,
        hovermode='x unified'
    )
    
    return fig

def create_sentiment_trend_chart(sentiment_data, title="Sentiment Trend Analysis", height=400):
    """Create sentiment trend visualization"""
    
    if isinstance(sentiment_data, dict):
        # Convert dict to DataFrame
        df = pd.DataFrame([
            {'Date': pd.to_datetime(date), 'Sentiment': sentiment}
            for date, sentiment in sentiment_data.items()
        ])
    else:
        df = sentiment_data.copy()
    
    fig = px.line(
        df,
        x='Date',
        y='Sentiment',
        title=title,
        markers=True
    )
    
    # Add neutral line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Neutral", annotation_position="bottom right")
    
    # Color the line based on sentiment
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8)
    )
    
    fig.update_layout(
        height=height,
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1, 1])
    )
    
    return fig

def create_revenue_waterfall(baseline, improvements, title="Revenue Impact Waterfall"):
    """Create waterfall chart showing revenue improvements"""
    
    categories = ['Baseline'] + list(improvements.keys()) + ['Total']
    values = [baseline] + list(improvements.values())
    cumulative = [baseline]
    
    for value in improvements.values():
        cumulative.append(cumulative[-1] + value)
    
    values.append(cumulative[-1])
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Revenue Impact",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(improvements) + ["total"],
        x=categories,
        textposition="outside",
        text=[f"₹{v:,.0f}" for v in values],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        yaxis_title="Revenue (₹)",
        height=400
    )
    
    return fig

def create_kpi_dashboard(kpi_data, height=600):
    """Create comprehensive KPI dashboard with multiple visualizations"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Metrics', 'NPS Progress', 'Efficiency Metrics', 'Volume Metrics'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Revenue indicator
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=kpi_data.get('current_revenue', 0),
            delta={'reference': kpi_data.get('target_revenue', 0)},
            gauge={'axis': {'range': [None, kpi_data.get('target_revenue', 500)]},
                  'bar': {'color': "darkblue"},
                  'steps': [{'range': [0, kpi_data.get('target_revenue', 500) * 0.8], 'color': "lightgray"}],
                  'threshold': {'line': {'color': "red", 'width': 4},
                              'thickness': 0.75, 'value': kpi_data.get('target_revenue', 500)}},
            title={'text': "Revenue/Pax (₹)"}
        ),
        row=1, col=1
    )
    
    # NPS indicator
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=kpi_data.get('current_nps', 0),
            delta={'reference': kpi_data.get('target_nps', 0)},
            gauge={'axis': {'range': [-100, 100]},
                  'bar': {'color': "darkgreen"},
                  'steps': [{'range': [-100, 0], 'color': "lightcoral"},
                           {'range': [0, 50], 'color': "lightyellow"},
                           {'range': [50, 100], 'color': "lightgreen"}]},
            title={'text': "Net Promoter Score"}
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=height, showlegend=False)
    return fig

def create_airport_comparison_radar(del_data, jai_data, categories):
    """Create radar chart comparing two airports"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=del_data,
        theta=categories,
        fill='toself',
        name='Delhi (DEL)',
        line_color='blue'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=jai_data,
        theta=categories,
        fill='toself',
        name='Jaipur (JAI)',
        line_color='orange'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Airport Performance Comparison",
        height=500
    )
    
    return fig

def create_heatmap(data, x_col, y_col, value_col, title="Heatmap"):
    """Create heatmap visualization"""
    
    # Pivot data for heatmap
    pivot_data = data.pivot(index=y_col, columns=x_col, values=value_col)
    
    fig = px.imshow(
        pivot_data,
        title=title,
        color_continuous_scale='RdYlBu_r',
        aspect='auto'
    )
    
    fig.update_layout(height=400)
    return fig

def create_funnel_chart(stages, values, title="Conversion Funnel"):
    """Create funnel chart for conversion analysis"""
    
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textinfo="value+percent initial"
    ))
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    return fig

def create_time_series_forecast(historical_data, forecast_data, 
                               title="Time Series with Forecast", height=400):
    """Create time series visualization with forecast"""
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data['value'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast data
    if forecast_data is not None and not forecast_data.empty:
        fig.add_trace(go.Scatter(
            x=forecast_data['date'],
            y=forecast_data['value'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        # Add confidence intervals if available
        if 'upper_bound' in forecast_data.columns and 'lower_bound' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data['date'].tolist() + forecast_data['date'].tolist()[::-1],
                y=forecast_data['upper_bound'].tolist() + forecast_data['lower_bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Confidence Interval'
            ))
    
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Date",
        yaxis_title="Value"
    )
    
    return fig

# Color schemes and themes
AIRPORT_COLORS = {
    'DEL': '#1f77b4',  # Blue
    'JAI': '#ff7f0e',  # Orange
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8'
}

SENTIMENT_COLORS = {
    'positive': '#28a745',
    'negative': '#dc3545', 
    'neutral': '#6c757d'
}

NPS_COLORS = {
    'promoter': '#28a745',
    'passive': '#ffc107',
    'detractor': '#dc3545'
}
