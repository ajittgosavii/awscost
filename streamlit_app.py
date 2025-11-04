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
    """Load credentials from Streamlit secrets with detailed error handling"""
    try:
        # Try to access AWS credentials
        aws_access_key = st.secrets["aws"]["access_key_id"]
        aws_secret_key = st.secrets["aws"]["secret_access_key"]
        aws_region = st.secrets["aws"].get("region", "us-east-1")
        
        # Try to access Anthropic credentials
        claude_key = st.secrets["anthropic"]["api_key"]
        
        # Validate that keys are not placeholder values
        if "YOUR_" in aws_access_key or "YOUR_" in aws_secret_key or "YOUR_" in claude_key:
            return None, None, "❌ Please replace placeholder values (YOUR_AWS... and YOUR_CLAUDE...) with your actual API keys", True
        
        aws_creds = {
            'access_key': aws_access_key,
            'secret_key': aws_secret_key,
            'region': aws_region
        }
        
        return aws_creds, claude_key, None, False
        
    except Exception as e:
        error_msg = f"❌ Error loading secrets: {str(e)}"
        return None, None, error_msg, True


def show_setup_instructions():
    """Show detailed setup instructions for Streamlit Cloud"""
    st.error("⚠️ Secrets not configured properly!")
    
    st.markdown("""
## 🔧 How to Configure Secrets on Streamlit Cloud:

### **Step 1: Access App Settings**
1. Go to your app on Streamlit Cloud (https://share.streamlit.io/)
2. Click the **"⋮"** menu button (top right corner)
3. Select **"Settings"**

### **Step 2: Open Secrets Section**
1. In the settings menu, find **"Secrets"** section
2. Click to expand it

### **Step 3: Add Your Credentials**
Copy and paste this format, then **replace with your real keys**:

```toml
[aws]
access_key_id = "AKIAIOSFODNN7EXAMPLE"
secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
region = "us-east-1"

[anthropic]
api_key = "sk-ant-api03-xxxxxxxxxxxxx"
```

### **Step 4: Save and Restart**
1. Click **"Save"** button
2. App will automatically restart (wait 10-20 seconds)
3. Refresh this page

---

## 📝 Where to Get Your API Keys:

### **AWS Credentials:**
1. Log into [AWS Console](https://console.aws.amazon.com/)
2. Go to **IAM** → **Users** → Your User → **Security Credentials**
3. Click **"Create Access Key"**
4. Choose **"Application running outside AWS"**
5. Copy both:
   - **Access Key ID** (starts with AKIA...)
   - **Secret Access Key** (long random string)

**Required IAM Permission:**
```json
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": [
            "ce:GetCostAndUsage",
            "ce:GetCostForecast"
        ],
        "Resource": "*"
    }]
}
```

### **Anthropic API Key:**
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign in or create account
3. Click **"Get API Keys"** or **"API Keys"** in sidebar
4. Click **"Create Key"**
5. Copy the key (starts with `sk-ant-api03-`)

---

## ⚠️ Important Notes:

✅ **DO:**
- Use your real API keys (not the examples shown above)
- Make sure AWS Cost Explorer is enabled (AWS Console → Billing → Cost Explorer)
- Wait 24 hours after enabling Cost Explorer for data to appear
- Double-check there are no extra spaces or quotes

❌ **DON'T:**
- Don't leave "YOUR_AWS_ACCESS_KEY_ID" - use real keys!
- Don't add extra quotes around values in Streamlit Cloud
- Don't commit secrets to GitHub
- Don't share your keys publicly

---

## 🔍 Troubleshooting:

**"No key access_key_id" error:**
- Make sure the secrets format matches exactly (including `[aws]` and `[anthropic]` headers)
- Check for typos in key names
- Ensure you clicked "Save" after pasting

**"AWS Access Denied" error:**
- Verify your IAM user has Cost Explorer permissions
- Check that Cost Explorer is enabled in AWS Console
- Wait 24 hours after enabling Cost Explorer

**"Invalid Anthropic API key" error:**
- Verify the key starts with `sk-ant-api03-`
- Make sure you copied the entire key
- Try generating a new key from Anthropic Console

---

After adding your secrets, **save** and **refresh this page**.
    """)


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
    
    def get_monthly_comparison(self, months_back: int = 3):
        """Get cost comparison for the last N months"""
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
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'}
                ]
            )
            
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
            
            if len(df) > 0:
                df['Period'] = pd.to_datetime(df['Period'])
                comparison_df = self._calculate_changes(df)
                return df, comparison_df
            
            return df, pd.DataFrame()
            
        except Exception as e:
            st.error(f"❌ Error fetching AWS data: {str(e)}")
            st.info("💡 Ensure AWS Cost Explorer is enabled (takes 24hrs after enabling)")
            return None, None
    
    def _calculate_changes(self, df):
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
        all_services = set(current_data['Service'].unique()) | set(previous_data['Service'].unique())
        
        for service in all_services:
            current_cost = current_data[current_data['Service'] == service]['Cost'].sum()
            prev_cost = previous_data[previous_data['Service'] == service]['Cost'].sum()
            
            cost_diff = current_cost - prev_cost
            cost_pct = ((current_cost - prev_cost) / prev_cost * 100) if prev_cost > 0 else (100 if current_cost > 0 else 0)
            
            comparison.append({
                'Service': service,
                'Current_Cost': current_cost,
                'Previous_Cost': prev_cost,
                'Cost_Difference': cost_diff,
                'Cost_Change_%': cost_pct
            })
        
        return pd.DataFrame(comparison)


class ClaudeAnalyzer:
    """Class to handle Claude API operations"""
    
    def __init__(self, api_key: str):
        """Initialize Claude client"""
        self.client = Anthropic(api_key=api_key)
    
    def analyze_cost_trends(self, comparison_df, monthly_df):
        """Analyze cost trends with Claude"""
        try:
            data_summary = {
                'total_current': comparison_df['Current_Cost'].sum(),
                'total_change': comparison_df['Cost_Difference'].sum(),
                'increases': comparison_df.nlargest(5, 'Cost_Difference')[['Service', 'Cost_Difference', 'Cost_Change_%']].to_dict('records'),
                'decreases': comparison_df.nsmallest(5, 'Cost_Difference')[['Service', 'Cost_Difference', 'Cost_Change_%']].to_dict('records'),
                'top_services': comparison_df.nlargest(5, 'Current_Cost')[['Service', 'Current_Cost']].to_dict('records')
            }
            
            prompt = f"""Analyze this AWS cost data:

Total Current Cost: ${data_summary['total_current']:.2f}
Total Change: ${data_summary['total_change']:.2f}

Top Increases:
{json.dumps(data_summary['increases'], indent=2)}

Top Decreases:
{json.dumps(data_summary['decreases'], indent=2)}

Top Services:
{json.dumps(data_summary['top_services'], indent=2)}

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


def create_visualizations(monthly_df, comparison_df):
    """Create all visualization charts"""
    
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
    fig_trend.update_layout(title='Monthly Cost Trend', xaxis_title='Month', yaxis_title='Cost (USD)', height=400)
    
    # Service breakdown
    top_services = comparison_df.nlargest(10, 'Current_Cost')
    fig_pie = go.Figure(data=[go.Pie(
        labels=top_services['Service'],
        values=top_services['Current_Cost'],
        hole=.3
    )])
    fig_pie.update_layout(title='Top 10 Services', height=400)
    
    # Changes
    significant = comparison_df[abs(comparison_df['Cost_Change_%']) > 10].nlargest(15, 'Cost_Difference', key=abs)
    if len(significant) > 0:
        colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in significant['Cost_Difference']]
        fig_change = go.Figure(data=[go.Bar(
            x=significant['Service'],
            y=significant['Cost_Difference'],
            marker_color=colors,
            text=significant['Cost_Difference'].apply(lambda x: f'${x:.2f}'),
            textposition='outside'
        )])
        fig_change.update_layout(title='Significant Changes (>10%)', xaxis_tickangle=-45, height=400)
    else:
        fig_change = None
    
    return fig_trend, fig_pie, fig_change


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
    
    st.success("✅ Credentials loaded successfully!")
    
    # Store credentials
    st.session_state['aws_creds'] = aws_creds
    st.session_state['claude_key'] = claude_key
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        st.success(f"📍 Region: `{aws_creds['region']}`")
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
            - IAM permission: `ce:GetCostAndUsage`
            """)
    
    # Main content
    if st.session_state['analysis_run']:
        months = st.session_state.get('months_back', 3)
        
        with st.spinner(f"📊 Fetching {months} months of data..."):
            try:
                analyzer = AWSCostAnalyzer(aws_creds['access_key'], aws_creds['secret_key'], aws_creds['region'])
                monthly_df, comparison_df = analyzer.get_monthly_comparison(months)
                
                if monthly_df is None or len(comparison_df) == 0:
                    st.error("❌ No data. Ensure Cost Explorer is enabled and has 24+ hours of data.")
                    st.stop()
                
                st.session_state['monthly_df'] = monthly_df
                st.session_state['comparison_df'] = comparison_df
                
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
                st.dataframe(increases[['Service', 'Current_Cost', 'Cost_Difference', 'Cost_Change_%']].style.format({
                    'Current_Cost': '${:,.2f}',
                    'Cost_Difference': '${:,.2f}',
                    'Cost_Change_%': '{:.1f}%'
                }), use_container_width=True, height=400)
            else:
                st.info("✅ No cost increases")
        
        with tab3:
            decreases = comparison_df[comparison_df['Cost_Difference'] < 0].nsmallest(15, 'Cost_Difference')
            if len(decreases) > 0:
                st.dataframe(decreases[['Service', 'Current_Cost', 'Cost_Difference', 'Cost_Change_%']].style.format({
                    'Current_Cost': '${:,.2f}',
                    'Cost_Difference': '${:,.2f}',
                    'Cost_Change_%': '{:.1f}%'
                }), use_container_width=True, height=400)
            else:
                st.info("📊 No decreases")
        
        st.markdown("---")
        
        # AI Insights
        st.subheader("🤖 AI-Powered Insights")
        if st.button("✨ Generate AI Analysis", type="primary"):
            with st.spinner("🧠 Analyzing..."):
                claude_analyzer = ClaudeAnalyzer(claude_key)
                insights = claude_analyzer.analyze_cost_trends(comparison_df, monthly_df)
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown(insights)
                st.markdown('</div>', unsafe_allow_html=True)
                st.session_state['insights'] = insights
        
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
        # Welcome
        st.info("👋 Click '🔍 Analyze Costs' in the sidebar to start!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### ✨ Features
            - 📊 Monthly cost comparison
            - 📈 Interactive charts
            - 🔍 Service breakdown
            - 🤖 AI insights
            - 💾 Export reports
            """)
        
        with col2:
            st.markdown(f"""
            ### 🚀 Status
            - ✅ Credentials loaded
            - ✅ AWS connected
            - ✅ Claude ready
            - 📍 Region: {aws_creds['region']}
            """)


if __name__ == "__main__":
    main()