"""
AWS Cost Analyzer with AI Insights
Single-file Streamlit application for Streamlit Cloud deployment
Uses Streamlit secrets for secure credential management
"""

import streamlit as st
import boto3
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from anthropic import Anthropic
import json
from typing import Dict, List, Tuple

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
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class AWSCostAnalyzer:
    """Class to handle AWS Cost Explorer operations"""
    
    def __init__(self, access_key: str, secret_key: str, region: str = 'us-east-1'):
        """Initialize AWS Cost Explorer client"""
        self.ce_client = boto3.client(
            'ce',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
    
    def get_cost_and_usage(self, start_date: str, end_date: str, granularity: str = 'MONTHLY') -> Dict:
        """Fetch cost and usage data from AWS Cost Explorer"""
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity=granularity,
                Metrics=['UnblendedCost', 'UsageQuantity'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                ]
            )
            return response
        except Exception as e:
            st.error(f"Error fetching AWS cost data: {str(e)}")
            return None
    
    def get_monthly_comparison(self, months_back: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get cost comparison for the last N months"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=months_back * 30)
        
        response = self.get_cost_and_usage(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            'MONTHLY'
        )
        
        if not response:
            return None, None
        
        # Parse the response
        monthly_data = []
        for result in response['ResultsByTime']:
            period_start = result['TimePeriod']['Start']
            for group in result['Groups']:
                service = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                usage = float(group['Metrics']['UsageQuantity']['Amount'])
                
                monthly_data.append({
                    'Period': period_start,
                    'Service': service,
                    'Cost': cost,
                    'Usage': usage
                })
        
        df = pd.DataFrame(monthly_data)
        
        # Calculate month-over-month changes
        if len(df) > 0:
            df['Period'] = pd.to_datetime(df['Period'])
            comparison_df = self._calculate_changes(df)
            return df, comparison_df
        
        return df, pd.DataFrame()
    
    def _calculate_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate month-over-month changes"""
        df_sorted = df.sort_values('Period')
        periods = df_sorted['Period'].unique()
        
        if len(periods) < 2:
            return pd.DataFrame()
        
        current_period = periods[-1]
        previous_period = periods[-2]
        
        current_data = df_sorted[df_sorted['Period'] == current_period]
        previous_data = df_sorted[df_sorted['Period'] == previous_period]
        
        comparison = []
        
        # Get all services from both periods
        all_services = set(current_data['Service'].unique()) | set(previous_data['Service'].unique())
        
        for service in all_services:
            current_cost = current_data[current_data['Service'] == service]['Cost'].sum()
            current_usage = current_data[current_data['Service'] == service]['Usage'].sum()
            
            prev_cost = previous_data[previous_data['Service'] == service]['Cost'].sum()
            prev_usage = previous_data[previous_data['Service'] == service]['Usage'].sum()
            
            cost_diff = current_cost - prev_cost
            cost_pct_change = ((current_cost - prev_cost) / prev_cost * 100) if prev_cost > 0 else (100 if current_cost > 0 else 0)
            
            usage_diff = current_usage - prev_usage
            usage_pct_change = ((current_usage - prev_usage) / prev_usage * 100) if prev_usage > 0 else (100 if current_usage > 0 else 0)
            
            comparison.append({
                'Service': service,
                'Current_Cost': current_cost,
                'Previous_Cost': prev_cost,
                'Cost_Difference': cost_diff,
                'Cost_Change_%': cost_pct_change,
                'Current_Usage': current_usage,
                'Previous_Usage': prev_usage,
                'Usage_Difference': usage_diff,
                'Usage_Change_%': usage_pct_change
            })
        
        return pd.DataFrame(comparison)


class ClaudeAnalyzer:
    """Class to handle Claude API operations for intelligent analysis"""
    
    def __init__(self, api_key: str):
        """Initialize Anthropic Claude client"""
        self.client = Anthropic(api_key=api_key)
    
    def analyze_cost_trends(self, comparison_df: pd.DataFrame, monthly_df: pd.DataFrame) -> str:
        """Use Claude to analyze cost trends and provide insights"""
        
        # Prepare data summary for Claude
        data_summary = {
            'top_cost_increases': comparison_df.nlargest(5, 'Cost_Difference')[
                ['Service', 'Cost_Difference', 'Cost_Change_%']
            ].to_dict('records'),
            'top_cost_decreases': comparison_df.nsmallest(5, 'Cost_Difference')[
                ['Service', 'Cost_Difference', 'Cost_Change_%']
            ].to_dict('records'),
            'total_cost_change': comparison_df['Cost_Difference'].sum(),
            'total_current_cost': comparison_df['Current_Cost'].sum(),
            'services_with_increases': len(comparison_df[comparison_df['Cost_Difference'] > 0]),
            'services_with_decreases': len(comparison_df[comparison_df['Cost_Difference'] < 0]),
            'top_services': comparison_df.nlargest(5, 'Current_Cost')[['Service', 'Current_Cost']].to_dict('records')
        }
        
        prompt = f"""Analyze the following AWS cost data and provide actionable insights:

Monthly Cost Comparison Data:
{json.dumps(data_summary, indent=2, default=str)}

Please provide a comprehensive analysis with:

1. **Executive Summary**: Brief overview of the cost situation (2-3 sentences)

2. **Key Findings**: 
   - Most significant cost changes
   - Overall spending trend
   - Notable patterns

3. **Cost Drivers**: 
   - Which services are driving cost increases
   - Why these increases might be occurring

4. **Optimization Opportunities**: 
   - Specific areas where costs could be reduced
   - Quick wins vs. long-term optimizations

5. **Anomalies & Concerns**: 
   - Any unusual patterns or unexpected changes
   - Services that require immediate attention

6. **Actionable Recommendations**: 
   - 5 specific, prioritized actions to take
   - Expected impact of each recommendation

Format your response with clear markdown headers and bullet points for readability."""
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=3000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error generating Claude analysis: {str(e)}"
    
    def get_service_recommendations(self, service_name: str, cost_change: float, usage_change: float) -> str:
        """Get specific recommendations for a service"""
        
        prompt = f"""Provide specific cost optimization recommendations for AWS service: {service_name}

Current situation:
- Cost change: ${cost_change:.2f} ({((cost_change / abs(cost_change)) * 100) if cost_change != 0 else 0:.1f}%)
- Usage change: {usage_change:.1f}%

Please provide:
1. Why this cost change might be occurring
2. 3-4 specific optimization strategies for this service
3. Potential risks or considerations
4. Tools or AWS features that can help

Keep recommendations practical and actionable."""
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"


def get_credentials():
    """Get credentials from Streamlit secrets"""
    try:
        aws_credentials = {
            'access_key_id': st.secrets["aws"]["access_key_id"],
            'secret_access_key': st.secrets["aws"]["secret_access_key"],
            'region': st.secrets["aws"].get("region", "us-east-1")
        }
        
        anthropic_key = st.secrets["anthropic"]["api_key"]
        
        return aws_credentials, anthropic_key, True
        
    except Exception as e:
        st.error(f"❌ Error loading secrets: {str(e)}")
        st.error("Please configure secrets in Streamlit Cloud settings.")
        st.info("""
        **Required secrets format:**
        ```toml
        [aws]
        access_key_id = "YOUR_AWS_ACCESS_KEY_ID"
        secret_access_key = "YOUR_AWS_SECRET_ACCESS_KEY"
        region = "us-east-1"

        [anthropic]
        api_key = "YOUR_CLAUDE_API_KEY"
        ```
        """)
        return None, None, False


def main():
    """Main application function"""
    st.markdown('<p class="main-header">💰 AWS Cost Analyzer with AI Insights</p>', unsafe_allow_html=True)
    
    # Load credentials from secrets
    aws_creds, claude_key, creds_loaded = get_credentials()
    
    if not creds_loaded:
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.success("✅ Credentials loaded from secrets")
        
        # Show configured region
        st.info(f"📍 Region: {aws_creds['region']}")
        
        # Analysis period
        st.subheader("Analysis Period")
        months_back = st.slider("Months to analyze", 2, 12, 3, help="Select how many months of historical data to analyze")
        
        analyze_button = st.button("🔍 Analyze Costs", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ Help & Information"):
            st.markdown("""
            **How to use:**
            1. Select the number of months to analyze
            2. Click "Analyze Costs" to fetch data
            3. Review metrics and visualizations
            4. Generate AI insights for recommendations
            5. Export data as needed
            
            **Requirements:**
            - AWS Cost Explorer must be enabled
            - Wait 24 hours after enabling for data
            - IAM user needs Cost Explorer permissions
            """)
    
    # Main content
    if analyze_button:
        with st.spinner("🔄 Fetching AWS cost data..."):
            try:
                # Initialize analyzer
                analyzer = AWSCostAnalyzer(
                    aws_creds['access_key_id'],
                    aws_creds['secret_access_key'],
                    aws_creds['region']
                )
                
                # Fetch data
                monthly_df, comparison_df = analyzer.get_monthly_comparison(months_back)
                
                if monthly_df is None or len(monthly_df) == 0:
                    st.error("❌ No cost data available for the selected period")
                    st.info("""
                    **Possible reasons:**
                    - Cost Explorer was recently enabled (wait 24 hours)
                    - No AWS resources are running
                    - Date range is invalid
                    - IAM permissions are insufficient
                    """)
                    return
                
                # Store in session state
                st.session_state['monthly_df'] = monthly_df
                st.session_state['comparison_df'] = comparison_df
                st.session_state['analyzer'] = analyzer
                st.session_state['claude_key'] = claude_key
                
                st.success("✅ Cost data loaded successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error("Please verify your AWS credentials and permissions.")
                with st.expander("🔍 Troubleshooting"):
                    st.markdown("""
                    **Common issues:**
                    1. **Invalid credentials**: Check secrets configuration
                    2. **Permission denied**: Verify IAM policy includes `ce:GetCostAndUsage`
                    3. **Cost Explorer not enabled**: Enable in AWS Billing Console
                    4. **No data**: Wait 24 hours after enabling Cost Explorer
                    """)
                return
    
    # Display results if data is available
    if 'comparison_df' in st.session_state and len(st.session_state['comparison_df']) > 0:
        comparison_df = st.session_state['comparison_df']
        monthly_df = st.session_state['monthly_df']
        
        # Key Metrics
        st.header("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_current = comparison_df['Current_Cost'].sum()
        total_previous = comparison_df['Previous_Cost'].sum()
        total_change = comparison_df['Cost_Difference'].sum()
        pct_change = (total_change / total_previous * 100) if total_previous > 0 else 0
        
        with col1:
            st.metric("Current Month Cost", f"${total_current:,.2f}")
        with col2:
            st.metric("Previous Month Cost", f"${total_previous:,.2f}")
        with col3:
            st.metric("Cost Change", f"${total_change:,.2f}", f"{pct_change:.1f}%")
        with col4:
            services_changed = len(comparison_df[comparison_df['Cost_Difference'] != 0])
            st.metric("Services Changed", services_changed)
        
        # Cost Trend Chart
        st.header("📈 Cost Trends Over Time")
        monthly_summary = monthly_df.groupby('Period')['Cost'].sum().reset_index()
        monthly_summary['Period'] = pd.to_datetime(monthly_summary['Period'])
        monthly_summary = monthly_summary.sort_values('Period')
        
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
            yaxis_title='Cost ($)',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Service Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of current costs
            top_services = comparison_df.nlargest(10, 'Current_Cost')
            fig_pie = px.pie(
                top_services,
                values='Current_Cost',
                names='Service',
                title='Top 10 Services by Cost Distribution'
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart of cost changes
            top_changes = comparison_df.nlargest(10, 'Cost_Difference', keep='all')
            top_changes = top_changes.sort_values('Cost_Difference', ascending=True)
            
            fig_changes = go.Figure()
            colors = ['red' if x > 0 else 'green' for x in top_changes['Cost_Difference']]
            fig_changes.add_trace(go.Bar(
                y=top_changes['Service'],
                x=top_changes['Cost_Difference'],
                orientation='h',
                marker_color=colors,
                text=top_changes['Cost_Difference'].apply(lambda x: f'${x:,.2f}'),
                textposition='outside'
            ))
            fig_changes.update_layout(
                title='Top 10 Cost Changes',
                xaxis_title='Cost Difference ($)',
                yaxis_title='Service',
                height=400
            )
            st.plotly_chart(fig_changes, use_container_width=True)
        
        # Service-wise breakdown
        st.header("🔍 Service-wise Cost Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📊 All Services", "📈 Increases", "📉 Decreases"])
        
        with tab1:
            # Searchable and sortable table
            st.dataframe(
                comparison_df.sort_values('Current_Cost', ascending=False).style.format({
                    'Current_Cost': '${:,.2f}',
                    'Previous_Cost': '${:,.2f}',
                    'Cost_Difference': '${:,.2f}',
                    'Cost_Change_%': '{:.1f}%',
                    'Current_Usage': '{:,.2f}',
                    'Previous_Usage': '{:,.2f}',
                    'Usage_Difference': '{:,.2f}',
                    'Usage_Change_%': '{:.1f}%'
                }).background_gradient(subset=['Cost_Difference'], cmap='RdYlGn_r'),
                use_container_width=True,
                height=400
            )
        
        with tab2:
            increases = comparison_df[comparison_df['Cost_Difference'] > 0].nlargest(15, 'Cost_Difference')
            if len(increases) > 0:
                st.dataframe(
                    increases[['Service', 'Current_Cost', 'Previous_Cost', 'Cost_Difference', 'Cost_Change_%']].style.format({
                        'Current_Cost': '${:,.2f}',
                        'Previous_Cost': '${:,.2f}',
                        'Cost_Difference': '${:,.2f}',
                        'Cost_Change_%': '{:.1f}%'
                    }).background_gradient(subset=['Cost_Difference'], cmap='Reds'),
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
            else:
                st.info("✅ No cost increases found")
        
        with tab3:
            decreases = comparison_df[comparison_df['Cost_Difference'] < 0].nsmallest(15, 'Cost_Difference')
            if len(decreases) > 0:
                st.dataframe(
                    decreases[['Service', 'Current_Cost', 'Previous_Cost', 'Cost_Difference', 'Cost_Change_%']].style.format({
                        'Current_Cost': '${:,.2f}',
                        'Previous_Cost': '${:,.2f}',
                        'Cost_Difference': '${:,.2f}',
                        'Cost_Change_%': '{:.1f}%'
                    }).background_gradient(subset=['Cost_Difference'], cmap='Greens_r'),
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
            else:
                st.info("📊 No cost decreases found")
        
        # Claude AI Insights
        if 'claude_key' in st.session_state and st.session_state['claude_key']:
            st.header("🤖 AI-Powered Insights")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("Get intelligent analysis and recommendations from Claude AI")
            with col2:
                generate_insights = st.button("✨ Generate AI Analysis", type="primary", use_container_width=True)
            
            if generate_insights:
                with st.spinner("🧠 Claude is analyzing your cost data..."):
                    claude_analyzer = ClaudeAnalyzer(st.session_state['claude_key'])
                    insights = claude_analyzer.analyze_cost_trends(comparison_df, monthly_df)
                    
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown(insights)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Store insights in session state
                    st.session_state['insights'] = insights
            
            # Service-specific recommendations
            if 'insights' in st.session_state:
                st.subheader("🎯 Service-Specific Recommendations")
                
                # Get services with significant changes
                significant_changes = comparison_df[
                    (comparison_df['Cost_Difference'].abs() > 10) | 
                    (comparison_df['Cost_Change_%'].abs() > 20)
                ].nlargest(5, 'Cost_Difference', keep='all')
                
                if len(significant_changes) > 0:
                    selected_service = st.selectbox(
                        "Select a service for detailed recommendations:",
                        significant_changes['Service'].tolist()
                    )
                    
                    if st.button(f"Get Recommendations for {selected_service}", use_container_width=True):
                        service_data = comparison_df[comparison_df['Service'] == selected_service].iloc[0]
                        
                        with st.spinner(f"Analyzing {selected_service}..."):
                            claude_analyzer = ClaudeAnalyzer(st.session_state['claude_key'])
                            service_rec = claude_analyzer.get_service_recommendations(
                                selected_service,
                                service_data['Cost_Difference'],
                                service_data['Usage_Change_%']
                            )
                            
                            st.markdown(f"#### Recommendations for {selected_service}")
                            st.markdown(service_rec)
        
        # Download options
        st.header("💾 Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison (CSV)",
                data=csv,
                file_name=f"aws_cost_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            csv_monthly = monthly_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Monthly Data (CSV)",
                data=csv_monthly,
                file_name=f"aws_monthly_costs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            if 'insights' in st.session_state:
                st.download_button(
                    label="📥 Download AI Insights (TXT)",
                    data=st.session_state['insights'],
                    file_name=f"aws_cost_insights_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    else:
        # Welcome screen
        st.info("👋 Welcome! Click '🔍 Analyze Costs' in the sidebar to get started.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✨ Features
            
            **Cost Analysis**
            - 📊 Monthly cost comparison across all AWS services
            - 📈 Interactive trend visualizations
            - 🔍 Service-level breakdown with increases/decreases
            
            **AI Insights**
            - 🤖 Intelligent cost analysis from Claude AI
            - 🎯 Service-specific optimization recommendations
            - 💡 Actionable cost-saving strategies
            
            **Export & Share**
            - 💾 Download reports in CSV format
            - 📄 Export AI insights for presentations
            - 📊 Share with your team
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Getting Started
            
            **Prerequisites:**
            1. AWS Cost Explorer must be enabled
            2. Wait 24 hours after enabling for data
            3. IAM user needs Cost Explorer permissions
            
            **How to use:**
            1. Select analysis period (2-12 months)
            2. Click "🔍 Analyze Costs"
            3. Review metrics and charts
            4. Generate AI insights
            5. Export reports as needed
            
            **Support:**
            - Click ℹ️ Help in sidebar for more info
            - Review AWS IAM permissions if errors occur
            - Ensure Cost Explorer is enabled in AWS Console
            """)
        
        st.markdown("---")
        
        # Quick stats/info
        st.markdown("""
        ### 📋 System Status
        
        ✅ **Credentials**: Loaded from Streamlit secrets  
        ✅ **AWS Integration**: Ready  
        ✅ **Claude AI**: Ready  
        ✅ **Export Functions**: Available  
        
        **Current Configuration:**
        - AWS Region: `{}`
        - Analysis Options: 2-12 months
        - AI Model: Claude Sonnet 4.5
        """.format(aws_creds['region'] if aws_creds else 'Not configured'))


if __name__ == "__main__":
    main()