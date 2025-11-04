"""
AWS Cost Analyzer with AI Insights
Streamlit application for AWS cost analysis with Claude AI integration
"""

import streamlit as st
import boto3
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from anthropic import Anthropic
import json

# Page configuration
st.set_page_config(
    page_title="AWS Cost Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9900;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_credentials():
    """Load credentials from Streamlit secrets"""
    try:
        # Support both uppercase and lowercase keys
        try:
            aws_access_key = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
            aws_secret_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
            aws_region = st.secrets["aws"].get("AWS_DEFAULT_REGION", "us-east-1")
        except KeyError:
            aws_access_key = st.secrets["aws"]["access_key_id"]
            aws_secret_key = st.secrets["aws"]["secret_access_key"]
            aws_region = st.secrets["aws"].get("region", "us-east-1")
        
        try:
            claude_key = st.secrets["anthropic"]["ANTHROPIC_API_KEY"]
        except KeyError:
            claude_key = st.secrets["anthropic"]["api_key"]
        
        if "YOUR_" in aws_access_key or "YOUR_" in claude_key:
            return None, None, "❌ Replace placeholder values with real API keys", True
        
        aws_creds = {
            'access_key': aws_access_key,
            'secret_key': aws_secret_key,
            'region': aws_region
        }
        
        return aws_creds, claude_key, None, False
        
    except Exception as e:
        return None, None, f"❌ Error loading secrets: {str(e)}", True


def show_setup_instructions():
    """Show setup instructions"""
    st.error("⚠️ Secrets not configured properly!")
    st.markdown("""
## 🔧 Configure Secrets on Streamlit Cloud:

1. Go to your app → **Settings** (⋮ menu)
2. Open **Secrets** section
3. Paste this format with your real keys:

```toml
[aws]
AWS_ACCESS_KEY_ID = "YOUR_AWS_KEY"
AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET"
AWS_DEFAULT_REGION = "us-east-1"

[anthropic]
ANTHROPIC_API_KEY = "YOUR_CLAUDE_KEY"
```

4. Click **Save**
5. Wait 10-20 seconds for restart
    """)


class AWSCostAnalyzer:
    """AWS Cost Explorer operations"""
    
    def __init__(self, access_key: str, secret_key: str, region: str = 'us-east-1'):
        self.ce_client = boto3.client(
            'ce',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
    
    def get_monthly_comparison(self, months_back: int = 3):
        """Get monthly cost comparison"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=months_back * 30)
            
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost', 'UsageQuantity'],
                GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
            )
            
            monthly_data = []
            for result in response['ResultsByTime']:
                period = result['TimePeriod']['Start']
                for group in result['Groups']:
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    usage = float(group['Metrics']['UsageQuantity']['Amount'])
                    monthly_data.append({
                        'Period': period,
                        'Service': service,
                        'Cost': cost,
                        'Usage': usage
                    })
            
            df = pd.DataFrame(monthly_data)
            if len(df) > 0:
                df['Period'] = pd.to_datetime(df['Period'])
                comparison_df = self._calculate_changes(df)
                return df, comparison_df
            
            return df, pd.DataFrame()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("💡 Ensure Cost Explorer is enabled (takes 24hrs)")
            return None, None
    
    def get_regional_breakdown(self, months_back: int = 3):
        """Get cost breakdown by region"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=months_back * 30)
            
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                GroupBy=[{'Type': 'DIMENSION', 'Key': 'REGION'}]
            )
            
            regional_data = []
            for result in response['ResultsByTime']:
                period = result['TimePeriod']['Start']
                for group in result['Groups']:
                    region = group['Keys'][0]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    if cost > 0:
                        regional_data.append({
                            'Period': period,
                            'Region': region,
                            'Cost': cost
                        })
            
            return pd.DataFrame(regional_data)
            
        except Exception as e:
            st.warning(f"⚠️ Regional data unavailable: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_changes(self, df):
        """Calculate month-over-month changes"""
        df_sorted = df.sort_values('Period')
        periods = df_sorted['Period'].unique()
        
        if len(periods) < 2:
            return pd.DataFrame()
        
        current = periods[-1]
        previous = periods[-2]
        
        current_data = df_sorted[df_sorted['Period'] == current]
        previous_data = df_sorted[df_sorted['Period'] == previous]
        
        comparison = []
        all_services = set(current_data['Service'].unique()) | set(previous_data['Service'].unique())
        
        for service in all_services:
            curr_cost = current_data[current_data['Service'] == service]['Cost'].sum()
            prev_cost = previous_data[previous_data['Service'] == service]['Cost'].sum()
            
            diff = curr_cost - prev_cost
            pct = ((curr_cost - prev_cost) / prev_cost * 100) if prev_cost > 0 else (100 if curr_cost > 0 else 0)
            
            comparison.append({
                'Service': service,
                'Current_Cost': curr_cost,
                'Previous_Cost': prev_cost,
                'Cost_Difference': diff,
                'Cost_Change_%': pct
            })
        
        return pd.DataFrame(comparison)


class ClaudeAnalyzer:
    """Claude AI analysis"""
    
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def analyze_cost_trends(self, comparison_df, monthly_df):
        """Analyze cost trends with Claude"""
        try:
            summary = {
                'total_current': comparison_df['Current_Cost'].sum(),
                'total_change': comparison_df['Cost_Difference'].sum(),
                'increases': comparison_df.nlargest(5, 'Cost_Difference')[
                    ['Service', 'Cost_Difference', 'Cost_Change_%']
                ].to_dict('records'),
                'decreases': comparison_df.nsmallest(5, 'Cost_Difference')[
                    ['Service', 'Cost_Difference', 'Cost_Change_%']
                ].to_dict('records'),
                'top': comparison_df.nlargest(5, 'Current_Cost')[
                    ['Service', 'Current_Cost']
                ].to_dict('records')
            }
            
            prompt = f"""Analyze AWS cost data:

Total Current: ${summary['total_current']:.2f}
Change: ${summary['total_change']:.2f}

Top Increases: {json.dumps(summary['increases'], indent=2)}
Top Decreases: {json.dumps(summary['decreases'], indent=2)}
Top Services: {json.dumps(summary['top'], indent=2)}

Provide:
1. Executive Summary (2-3 sentences)
2. Key Findings (3-5 bullets)
3. Cost Drivers
4. Red Flags
5. 5 Optimization Recommendations
6. Quick Wins (2-3 items)"""

            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def get_service_recommendations(self, service: str, cost_change: float, usage_change: float):
        """Get service-specific recommendations"""
        try:
            prompt = f"""Optimization recommendations for {service}

Cost Change: ${cost_change:.2f}
Usage Change: {usage_change:.1f}%

Provide:
1. Service Overview
2. Cost Analysis
3. 3-5 Optimization Strategies
4. Best Practices
5. Monitoring Metrics"""

            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
        except Exception as e:
            return f"❌ Error: {str(e)}"


def create_visualizations(monthly_df, comparison_df):
    """Create visualization charts"""
    
    # Monthly trend
    monthly_summary = monthly_df.groupby('Period')['Cost'].sum().reset_index().sort_values('Period')
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_summary['Period'],
        y=monthly_summary['Cost'],
        mode='lines+markers',
        name='Total Cost',
        line=dict(color='#FF9900', width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(255, 153, 0, 0.1)'
    ))
    fig_trend.update_layout(
        title='Monthly Cost Trend',
        xaxis_title='Month',
        yaxis_title='Cost (USD)',
        height=400
    )
    
    # Service breakdown
    top_services = comparison_df.nlargest(10, 'Current_Cost')
    fig_pie = go.Figure(data=[go.Pie(
        labels=top_services['Service'],
        values=top_services['Current_Cost'],
        hole=.3
    )])
    fig_pie.update_layout(title='Top 10 Services', height=400)
    
    # Changes
    significant = comparison_df[abs(comparison_df['Cost_Change_%']) > 10].copy()
    if len(significant) > 0:
        significant['abs_diff'] = significant['Cost_Difference'].abs()
        significant = significant.nlargest(15, 'abs_diff')
        colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in significant['Cost_Difference']]
        
        fig_change = go.Figure(data=[go.Bar(
            x=significant['Service'],
            y=significant['Cost_Difference'],
            marker_color=colors,
            text=significant['Cost_Difference'].apply(lambda x: f'${x:.2f}'),
            textposition='outside'
        )])
        fig_change.update_layout(
            title='Significant Changes (>10%)',
            xaxis_tickangle=-45,
            height=400
        )
    else:
        fig_change = None
    
    return fig_trend, fig_pie, fig_change


def create_regional_charts(regional_df):
    """Create regional visualizations"""
    if regional_df.empty:
        return None, None
    
    regional_df['Period'] = pd.to_datetime(regional_df['Period'])
    current_period = regional_df['Period'].max()
    current_regional = regional_df[regional_df['Period'] == current_period].sort_values('Cost', ascending=False)
    
    # Pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=current_regional['Region'],
        values=current_regional['Cost'],
        hole=.3
    )])
    fig_pie.update_layout(title='Cost by Region (Current Month)', height=400)
    
    # Trend chart
    fig_trend = go.Figure()
    for region in regional_df['Region'].unique():
        region_data = regional_df[regional_df['Region'] == region].sort_values('Period')
        fig_trend.add_trace(go.Scatter(
            x=region_data['Period'],
            y=region_data['Cost'],
            mode='lines+markers',
            name=region
        ))
    
    fig_trend.update_layout(
        title='Regional Cost Trends',
        xaxis_title='Month',
        yaxis_title='Cost (USD)',
        height=400
    )
    
    return fig_pie, fig_trend


def main():
    """Main application"""
    
    # Initialize session
    if 'analysis_run' not in st.session_state:
        st.session_state['analysis_run'] = False
    
    # Header
    st.markdown('<h1 class="main-header">📊 AWS Cost Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Real-time AWS cost analysis with AI insights from Claude**")
    st.markdown("---")
    
    # Load credentials
    aws_creds, claude_key, error_msg, show_help = load_credentials()
    
    if show_help:
        show_setup_instructions()
        st.stop()
    
    if error_msg:
        st.error(error_msg)
        st.stop()
    
    st.success("✅ Credentials loaded!")
    
    # Store credentials
    st.session_state['aws_creds'] = aws_creds
    st.session_state['claude_key'] = claude_key
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        st.success(f"📍 Region: `{aws_creds['region']}`")
        st.info("Cost Explorer aggregates costs from **ALL AWS regions worldwide**")
        st.markdown("---")
        
        months_back = st.slider("Months to Analyze", 2, 12, 3)
        st.markdown("---")
        
        if st.button("🔍 Analyze Costs", type="primary", use_container_width=True):
            st.session_state['analysis_run'] = True
            st.session_state['months_back'] = months_back
        
        st.markdown("---")
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **Prerequisites:**
            - AWS Cost Explorer enabled
            - 24 hours of data available
            - IAM: `ce:GetCostAndUsage`
            
            **Regional Costs:**
            - us-east-1 is just the API endpoint
            - Costs from ALL regions are included
            - Regional breakdown shows per-region costs
            """)
    
    # Main content
    if st.session_state['analysis_run']:
        months = st.session_state.get('months_back', 3)
        
        with st.spinner(f"📊 Fetching {months} months of data..."):
            try:
                analyzer = AWSCostAnalyzer(
                    aws_creds['access_key'],
                    aws_creds['secret_key'],
                    aws_creds['region']
                )
                
                monthly_df, comparison_df = analyzer.get_monthly_comparison(months)
                
                if monthly_df is None or len(comparison_df) == 0:
                    st.error("❌ No data. Ensure Cost Explorer is enabled with 24+ hours of data.")
                    st.stop()
                
                regional_df = analyzer.get_regional_breakdown(months)
                
                st.session_state['monthly_df'] = monthly_df
                st.session_state['comparison_df'] = comparison_df
                st.session_state['regional_df'] = regional_df
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.stop()
        
        # Metrics
        total_current = comparison_df['Current_Cost'].sum()
        total_previous = comparison_df['Previous_Cost'].sum()
        total_change = comparison_df['Cost_Difference'].sum()
        change_pct = (total_change / total_previous * 100) if total_previous > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Month", f"${total_current:,.2f}")
        col2.metric("Previous Month", f"${total_previous:,.2f}")
        col3.metric("Change", f"${total_change:,.2f}", f"{change_pct:.1f}%")
        col4.metric("Services", len(comparison_df))
        
        st.markdown("---")
        
        # Visualizations
        fig_trend, fig_pie, fig_change = create_visualizations(monthly_df, comparison_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_trend, use_container_width=True)
        with col2:
            st.plotly_chart(fig_pie, use_container_width=True)
        
        if fig_change:
            st.plotly_chart(fig_change, use_container_width=True)
        
        st.markdown("---")
        
        # Regional Breakdown
        st.subheader("🌍 Regional Cost Breakdown")
        regional_df = st.session_state.get('regional_df', pd.DataFrame())
        
        if not regional_df.empty:
            fig_reg_pie, fig_reg_trend = create_regional_charts(regional_df)
            
            col1, col2 = st.columns(2)
            with col1:
                if fig_reg_pie:
                    st.plotly_chart(fig_reg_pie, use_container_width=True)
            with col2:
                if fig_reg_trend:
                    st.plotly_chart(fig_reg_trend, use_container_width=True)
            
            # Regional table
            regional_df_display = regional_df.copy()
            regional_df_display['Period'] = pd.to_datetime(regional_df_display['Period'])
            current_period = regional_df_display['Period'].max()
            current_regional = regional_df_display[
                regional_df_display['Period'] == current_period
            ].sort_values('Cost', ascending=False)
            
            st.write("**Current Month by Region:**")
            st.dataframe(
                current_regional[['Region', 'Cost']].style.format({'Cost': '${:,.2f}'}),
                use_container_width=True,
                height=300
            )
        else:
            st.info("ℹ️ No regional breakdown available")
        
        st.markdown("---")
        
        # Data tables
        st.subheader("📊 Cost Breakdown")
        tab1, tab2, tab3 = st.tabs(["All Services", "Increases", "Decreases"])
        
        with tab1:
            st.dataframe(
                comparison_df.sort_values('Current_Cost', ascending=False).style.format({
                    'Current_Cost': '${:,.2f}',
                    'Previous_Cost': '${:,.2f}',
                    'Cost_Difference': '${:,.2f}',
                    'Cost_Change_%': '{:.1f}%'
                }),
                use_container_width=True,
                height=400
            )
        
        with tab2:
            increases = comparison_df[comparison_df['Cost_Difference'] > 0].nlargest(15, 'Cost_Difference')
            if len(increases) > 0:
                st.dataframe(
                    increases[['Service', 'Current_Cost', 'Cost_Difference', 'Cost_Change_%']].style.format({
                        'Current_Cost': '${:,.2f}',
                        'Cost_Difference': '${:,.2f}',
                        'Cost_Change_%': '{:.1f}%'
                    }),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("✅ No increases")
        
        with tab3:
            decreases = comparison_df[comparison_df['Cost_Difference'] < 0].nsmallest(15, 'Cost_Difference')
            if len(decreases) > 0:
                st.dataframe(
                    decreases[['Service', 'Current_Cost', 'Cost_Difference', 'Cost_Change_%']].style.format({
                        'Current_Cost': '${:,.2f}',
                        'Cost_Difference': '${:,.2f}',
                        'Cost_Change_%': '{:.1f}%'
                    }),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("📊 No decreases")
        
        st.markdown("---")
        
        # AI Insights
        st.subheader("🤖 AI-Powered Insights")
        if st.button("✨ Generate AI Analysis", type="primary"):
            with st.spinner("🧠 Analyzing..."):
                claude = ClaudeAnalyzer(claude_key)
                insights = claude.analyze_cost_trends(comparison_df, monthly_df)
                
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown(insights)
                st.markdown('</div>', unsafe_allow_html=True)
                st.session_state['insights'] = insights
        
        # Service recommendations
        if 'insights' in st.session_state:
            st.markdown("---")
            st.subheader("🎯 Service Recommendations")
            
            # Create absolute value column for sorting
            comparison_df_sorted = comparison_df.copy()
            comparison_df_sorted['abs_diff'] = comparison_df_sorted['Cost_Difference'].abs()
            
            significant = comparison_df_sorted[
                (comparison_df_sorted['abs_diff'] > 10) | 
                (comparison_df_sorted['Cost_Change_%'].abs() > 20)
            ].nlargest(5, 'abs_diff')
            
            if len(significant) > 0:
                selected = st.selectbox(
                    "Select service:",
                    significant['Service'].tolist()
                )
                
                if st.button(f"Get Recommendations for {selected}", use_container_width=True):
                    service_data = comparison_df[comparison_df['Service'] == selected].iloc[0]
                    
                    with st.spinner(f"Analyzing {selected}..."):
                        claude = ClaudeAnalyzer(claude_key)
                        rec = claude.get_service_recommendations(
                            selected,
                            service_data['Cost_Difference'],
                            service_data['Cost_Change_%']
                        )
                        
                        st.markdown(f"#### {selected} Recommendations")
                        st.markdown(rec)
        
        # Export
        st.markdown("---")
        st.subheader("💾 Export")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📥 Comparison (CSV)",
                comparison_df.to_csv(index=False),
                f"costs_{datetime.now().strftime('%Y%m%d')}.csv",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                "📥 Monthly Data (CSV)",
                monthly_df.to_csv(index=False),
                f"monthly_{datetime.now().strftime('%Y%m%d')}.csv",
                use_container_width=True
            )
        
        with col3:
            if 'insights' in st.session_state:
                st.download_button(
                    "📥 AI Insights (TXT)",
                    st.session_state['insights'],
                    f"insights_{datetime.now().strftime('%Y%m%d')}.txt",
                    use_container_width=True
                )
    
    else:
        # Welcome screen
        st.info("👋 Click '🔍 Analyze Costs' in the sidebar!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✨ Features
            - 📊 Monthly cost comparison
            - 🌍 Regional breakdown
            - 📈 Interactive charts
            - 🤖 AI insights from Claude
            - 💾 Export reports
            """)
        
        with col2:
            st.markdown(f"""
            ### 🚀 Status
            - ✅ Credentials loaded
            - ✅ AWS connected
            - ✅ Claude ready
            - 📍 Region: {aws_creds['region']}
            - 🌍 Shows ALL regions
            """)


if __name__ == "__main__":
    main()