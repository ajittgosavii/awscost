"""
Enterprise AWS Cost Management Platform
Advanced multi-account cost analysis with AI-powered insights
"""

import streamlit as st
import boto3
import pandas as pd
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import plotly.express as px
from anthropic import Anthropic
import json
from typing import Dict, List, Optional, Tuple
import hashlib
import time
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="Enterprise AWS Cost Management",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #232F3E;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== AUTHENTICATION ====================

class AuthManager:
    """Simple authentication manager for enterprise users"""
    
    # In production, use database with hashed passwords
    USERS = {
        "admin": {
            "password": hashlib.sha256("admin123".encode()).hexdigest(),
            "role": "admin",
            "name": "Admin User",
            "accounts": ["all"]
        },
        "finance": {
            "password": hashlib.sha256("finance123".encode()).hexdigest(),
            "role": "finance",
            "name": "Finance Team",
            "accounts": ["production", "staging"]
        },
        "viewer": {
            "password": hashlib.sha256("viewer123".encode()).hexdigest(),
            "role": "viewer",
            "name": "Read Only User",
            "accounts": ["production"]
        }
    }
    
    @staticmethod
    def authenticate(username: str, password: str) -> Optional[Dict]:
        """Authenticate user"""
        if username in AuthManager.USERS:
            user = AuthManager.USERS[username]
            if hashlib.sha256(password.encode()).hexdigest() == user["password"]:
                return {
                    "username": username,
                    "role": user["role"],
                    "name": user["name"],
                    "accounts": user["accounts"]
                }
        return None
    
    @staticmethod
    def check_permission(user: Dict, action: str) -> bool:
        """Check if user has permission for action"""
        permissions = {
            "admin": ["view", "edit", "export", "configure", "budget"],
            "finance": ["view", "export", "budget"],
            "viewer": ["view"]
        }
        return action in permissions.get(user["role"], [])


def show_login_page():
    """Display login page"""
    st.markdown('<h1 class="main-header">🔐 Enterprise AWS Cost Management</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Secure Login Required</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                user = AuthManager.authenticate(username, password)
                if user:
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = user
                    st.session_state['login_time'] = datetime.now()
                    st.success(f"Welcome, {user['name']}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        with st.expander("ℹ️ Demo Credentials"):
            st.markdown("""
            **Admin Access:**
            - Username: `admin`
            - Password: `admin123`
            
            **Finance Team:**
            - Username: `finance`
            - Password: `finance123`
            
            **Read Only:**
            - Username: `viewer`
            - Password: `viewer123`
            """)


# ==================== AWS INTEGRATION ====================

def load_credentials():
    """Load AWS and Claude credentials"""
    try:
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
        
        aws_creds = {
            'access_key': aws_access_key,
            'secret_key': aws_secret_key,
            'region': aws_region
        }
        
        return aws_creds, claude_key, None
        
    except Exception as e:
        return None, None, f"❌ Error loading secrets: {str(e)}"


class EnterpriseAWSCostAnalyzer:
    """Enterprise-grade AWS Cost Explorer with advanced features"""
    
    def __init__(self, access_key: str, secret_key: str, region: str = 'us-east-1'):
        self.ce_client = boto3.client(
            'ce',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        self.organizations_client = None
        try:
            self.organizations_client = boto3.client(
                'organizations',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )
        except:
            pass
    
    def get_linked_accounts(self) -> List[Dict]:
        """Get all linked AWS accounts (for Organizations)"""
        accounts = []
        
        if self.organizations_client:
            try:
                response = self.organizations_client.list_accounts()
                for account in response.get('Accounts', []):
                    if account['Status'] == 'ACTIVE':
                        accounts.append({
                            'Id': account['Id'],
                            'Name': account['Name'],
                            'Email': account.get('Email', ''),
                            'Status': account['Status']
                        })
            except:
                pass
        
        return accounts
    
    def get_cost_by_account(self, start_date: str, end_date: str, granularity: str = 'MONTHLY'):
        """Get costs grouped by account"""
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity=granularity,
                Metrics=['UnblendedCost'],
                GroupBy=[{'Type': 'DIMENSION', 'Key': 'LINKED_ACCOUNT'}]
            )
            
            account_data = []
            for result in response['ResultsByTime']:
                period = result['TimePeriod']['Start']
                for group in result['Groups']:
                    account_id = group['Keys'][0]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    account_data.append({
                        'Period': period,
                        'AccountId': account_id,
                        'Cost': cost
                    })
            
            return pd.DataFrame(account_data)
        except Exception as e:
            st.error(f"Error fetching account costs: {str(e)}")
            return pd.DataFrame()
    
    def get_cost_by_tags(self, start_date: str, end_date: str, tag_key: str):
        """Get costs grouped by custom tags"""
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity='MONTHLY',
                Metrics=['UnblendedCost'],
                GroupBy=[{'Type': 'TAG', 'Key': tag_key}]
            )
            
            tag_data = []
            for result in response['ResultsByTime']:
                period = result['TimePeriod']['Start']
                for group in result['Groups']:
                    tag_value = group['Keys'][0].replace(f'{tag_key}$', '')
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    tag_data.append({
                        'Period': period,
                        'TagValue': tag_value,
                        'Cost': cost
                    })
            
            return pd.DataFrame(tag_data)
        except Exception as e:
            st.warning(f"Tag data unavailable: {str(e)}")
            return pd.DataFrame()
    
    def get_cost_forecast(self, start_date: str, end_date: str):
        """Get AWS cost forecast"""
        try:
            response = self.ce_client.get_cost_forecast(
                TimePeriod={'Start': start_date, 'End': end_date},
                Metric='UNBLENDED_COST',
                Granularity='MONTHLY'
            )
            
            return {
                'total': float(response.get('Total', {}).get('Amount', 0)),
                'period': response.get('ForecastResultsByTime', [])
            }
        except Exception as e:
            return None
    
    def get_ri_coverage(self, start_date: str, end_date: str):
        """Get Reserved Instance coverage"""
        try:
            response = self.ce_client.get_reservation_coverage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity='MONTHLY'
            )
            
            coverage_data = []
            for item in response.get('CoveragesByTime', []):
                period = item['TimePeriod']['Start']
                coverage = item.get('Total', {})
                coverage_data.append({
                    'Period': period,
                    'CoverageHours': float(coverage.get('CoverageHours', {}).get('CoverageHoursPercentage', 0)),
                    'OnDemandCost': float(coverage.get('OnDemandCost', 0)),
                    'TotalCost': float(coverage.get('TotalRunningHours', 0))
                })
            
            return pd.DataFrame(coverage_data)
        except Exception as e:
            return pd.DataFrame()
    
    def get_savings_plans_coverage(self, start_date: str, end_date: str):
        """Get Savings Plans coverage"""
        try:
            response = self.ce_client.get_savings_plans_coverage(
                TimePeriod={'Start': start_date, 'End': end_date},
                Granularity='MONTHLY'
            )
            
            sp_data = []
            for item in response.get('SavingsPlansCoverages', []):
                period = item['TimePeriod']['Start']
                coverage = item.get('Coverage', {})
                sp_data.append({
                    'Period': period,
                    'CoveragePercentage': float(coverage.get('CoveragePercentage', 0)),
                    'OnDemandCost': float(coverage.get('OnDemandCost', 0)),
                    'SpendCoveredBySavingsPlans': float(coverage.get('SpendCoveredBySavingsPlans', 0))
                })
            
            return pd.DataFrame(sp_data)
        except Exception as e:
            return pd.DataFrame()
    
    def get_cost_anomalies(self, start_date: str, end_date: str):
        """Detect cost anomalies"""
        try:
            # Get anomaly monitors
            monitors_response = self.ce_client.get_anomaly_monitors()
            
            if not monitors_response.get('AnomalyMonitors'):
                return []
            
            monitor_arn = monitors_response['AnomalyMonitors'][0]['MonitorArn']
            
            # Get anomalies
            response = self.ce_client.get_anomalies(
                DateInterval={'StartDate': start_date, 'EndDate': end_date},
                MonitorArn=monitor_arn
            )
            
            anomalies = []
            for anomaly in response.get('Anomalies', []):
                anomalies.append({
                    'Date': anomaly.get('AnomalyStartDate'),
                    'Impact': float(anomaly.get('Impact', {}).get('MaxImpact', 0)),
                    'Service': anomaly.get('RootCauses', [{}])[0].get('Service', 'Unknown'),
                    'Score': float(anomaly.get('AnomalyScore', {}).get('CurrentScore', 0))
                })
            
            return anomalies
        except:
            return []
    
    def get_monthly_comparison(self, months_back: int = 3):
        """Get standard monthly comparison"""
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
            st.error(f"Error: {str(e)}")
            return None, None
    
    def get_regional_breakdown(self, months_back: int = 3):
        """Get regional cost breakdown"""
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
        except:
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


# ==================== BUDGET & ALERTS ====================

class BudgetManager:
    """Manage budgets and alerts"""
    
    @staticmethod
    def check_budget_alerts(comparison_df: pd.DataFrame, budgets: Dict) -> List[Dict]:
        """Check if costs exceed budgets"""
        alerts = []
        
        total_cost = comparison_df['Current_Cost'].sum()
        
        if 'monthly_total' in budgets:
            budget = budgets['monthly_total']
            utilization = (total_cost / budget) * 100
            
            if utilization >= 100:
                alerts.append({
                    'type': 'critical',
                    'message': f'🚨 Budget exceeded! ${total_cost:,.2f} / ${budget:,.2f} ({utilization:.1f}%)',
                    'severity': 'high'
                })
            elif utilization >= 80:
                alerts.append({
                    'type': 'warning',
                    'message': f'⚠️ Budget at {utilization:.1f}%: ${total_cost:,.2f} / ${budget:,.2f}',
                    'severity': 'medium'
                })
        
        # Service-level budgets
        for service, budget in budgets.get('services', {}).items():
            service_cost = comparison_df[comparison_df['Service'] == service]['Current_Cost'].sum()
            if service_cost > 0:
                utilization = (service_cost / budget) * 100
                if utilization >= 90:
                    alerts.append({
                        'type': 'warning',
                        'message': f'⚠️ {service} at {utilization:.1f}% of budget',
                        'severity': 'medium'
                    })
        
        return alerts


# ==================== ADVANCED ANALYTICS ====================

class AdvancedAnalytics:
    """Advanced cost analytics and ML predictions"""
    
    @staticmethod
    def calculate_burn_rate(monthly_df: pd.DataFrame) -> Dict:
        """Calculate cost burn rate"""
        monthly_df['Period'] = pd.to_datetime(monthly_df['Period'])
        monthly_summary = monthly_df.groupby('Period')['Cost'].sum().sort_index()
        
        if len(monthly_summary) < 2:
            return {}
        
        recent_cost = monthly_summary.iloc[-1]
        previous_cost = monthly_summary.iloc[-2]
        
        daily_burn = recent_cost / 30  # Rough daily estimate
        monthly_growth = ((recent_cost - previous_cost) / previous_cost * 100) if previous_cost > 0 else 0
        
        return {
            'daily_burn_rate': daily_burn,
            'monthly_growth_rate': monthly_growth,
            'projected_month_end': recent_cost * (30 / datetime.now().day)
        }
    
    @staticmethod
    def year_over_year_comparison(monthly_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate year-over-year comparison"""
        monthly_df['Period'] = pd.to_datetime(monthly_df['Period'])
        monthly_df['Year'] = monthly_df['Period'].dt.year
        monthly_df['Month'] = monthly_df['Period'].dt.month
        
        yoy_data = monthly_df.groupby(['Year', 'Month'])['Cost'].sum().reset_index()
        yoy_data = yoy_data.sort_values(['Month', 'Year'])
        
        return yoy_data
    
    @staticmethod
    def identify_optimization_opportunities(comparison_df: pd.DataFrame) -> List[Dict]:
        """Identify cost optimization opportunities"""
        opportunities = []
        
        # High cost, low change - potential rightsizing
        stable_high_cost = comparison_df[
            (comparison_df['Current_Cost'] > comparison_df['Current_Cost'].quantile(0.75)) &
            (comparison_df['Cost_Change_%'].abs() < 5)
        ]
        
        for _, row in stable_high_cost.head(3).iterrows():
            opportunities.append({
                'type': 'rightsizing',
                'service': row['Service'],
                'cost': row['Current_Cost'],
                'description': f'High stable cost service - consider rightsizing'
            })
        
        # Rapid growth services
        rapid_growth = comparison_df[comparison_df['Cost_Change_%'] > 50].head(3)
        for _, row in rapid_growth.iterrows():
            opportunities.append({
                'type': 'growth_alert',
                'service': row['Service'],
                'growth': row['Cost_Change_%'],
                'description': f'Rapid cost growth - review usage patterns'
            })
        
        return opportunities


# ==================== VISUALIZATIONS ====================

def create_enterprise_dashboard_charts(monthly_df, comparison_df, regional_df=None):
    """Create comprehensive dashboard visualizations"""
    
    # 1. Cost Trend with Forecast
    monthly_summary = monthly_df.groupby('Period')['Cost'].sum().reset_index().sort_values('Period')
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=monthly_summary['Period'],
        y=monthly_summary['Cost'],
        mode='lines+markers',
        name='Actual Cost',
        line=dict(color='#FF9900', width=3),
        marker=dict(size=10),
        fill='tozeroy'
    ))
    
    fig_trend.update_layout(
        title='Monthly Cost Trend & Forecast',
        xaxis_title='Month',
        yaxis_title='Cost (USD)',
        height=400,
        hovermode='x unified'
    )
    
    # 2. Service Breakdown Treemap
    top_services = comparison_df.nlargest(15, 'Current_Cost')
    fig_treemap = px.treemap(
        top_services,
        path=['Service'],
        values='Current_Cost',
        title='Service Cost Distribution',
        color='Cost_Change_%',
        color_continuous_scale='RdYlGn_r',
        height=400
    )
    
    # 3. Cost Changes Waterfall
    top_changes = comparison_df.copy()
    top_changes['abs_diff'] = top_changes['Cost_Difference'].abs()
    top_changes = top_changes.nlargest(10, 'abs_diff')
    
    fig_waterfall = go.Figure(go.Waterfall(
        x=top_changes['Service'],
        y=top_changes['Cost_Difference'],
        text=top_changes['Cost_Difference'].apply(lambda x: f'${x:,.0f}'),
        textposition='outside',
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig_waterfall.update_layout(
        title='Top 10 Cost Changes (Waterfall)',
        showlegend=False,
        height=400,
        xaxis_tickangle=-45
    )
    
    # 4. Heatmap of service costs over time
    pivot_data = monthly_df.pivot_table(
        index='Service',
        columns='Period',
        values='Cost',
        fill_value=0
    )
    
    # Top 15 services only
    top_15_services = comparison_df.nlargest(15, 'Current_Cost')['Service'].tolist()
    pivot_data_filtered = pivot_data.loc[pivot_data.index.isin(top_15_services)]
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_data_filtered.values,
        x=pivot_data_filtered.columns,
        y=pivot_data_filtered.index,
        colorscale='Viridis',
        hovertemplate='Service: %{y}<br>Period: %{x}<br>Cost: $%{z:,.2f}<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        title='Service Cost Heatmap',
        height=500,
        xaxis_title='Period',
        yaxis_title='Service'
    )
    
    return fig_trend, fig_treemap, fig_waterfall, fig_heatmap


def create_regional_sankey(regional_df):
    """Create Sankey diagram for regional cost flow"""
    if regional_df.empty:
        return None
    
    regional_df['Period'] = pd.to_datetime(regional_df['Period'])
    current_period = regional_df['Period'].max()
    current_regional = regional_df[regional_df['Period'] == current_period]
    
    # Create Sankey
    regions = current_regional['Region'].unique()
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Total AWS Cost"] + list(regions),
            color="blue"
        ),
        link=dict(
            source=[0] * len(regions),
            target=list(range(1, len(regions) + 1)),
            value=current_regional['Cost'].tolist()
        )
    )])
    
    fig.update_layout(
        title="Regional Cost Flow",
        height=400
    )
    
    return fig


# ==================== MAIN APPLICATION ====================

def main():
    """Enterprise main application"""
    
    # Check authentication
    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        show_login_page()
        return
    
    user = st.session_state['user']
    
    # Header with user info
    col1, col2, col3 = st.columns([2, 4, 2])
    with col1:
        st.markdown(f"**Welcome, {user['name']}**")
        st.caption(f"Role: {user['role'].upper()}")
    with col2:
        st.markdown('<h1 class="main-header">💼 Enterprise AWS Cost Management</h1>', unsafe_allow_html=True)
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Load credentials
    aws_creds, claude_key, error = load_credentials()
    if error:
        st.error(error)
        return
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state['analyzer'] = EnterpriseAWSCostAnalyzer(
            aws_creds['access_key'],
            aws_creds['secret_key'],
            aws_creds['region']
        )
    
    analyzer = st.session_state['analyzer']
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("📊 Navigation")
        
        page = st.radio(
            "Select View",
            [
                "🏠 Executive Dashboard",
                "💰 Cost Analysis",
                "🌍 Multi-Account View",
                "🏷️ Tag-Based Analysis",
                "📊 RI & Savings Plans",
                "🚨 Anomaly Detection",
                "💡 Optimization Insights",
                "📈 Custom Reports",
                "⚙️ Settings & Budgets"
            ]
        )
        
        st.markdown("---")
        st.subheader("⚙️ Parameters")
        
        date_range = st.selectbox(
            "Date Range",
            ["Last 3 months", "Last 6 months", "Last 12 months", "Custom"]
        )
        
        months_map = {
            "Last 3 months": 3,
            "Last 6 months": 6,
            "Last 12 months": 12
        }
        months_back = months_map.get(date_range, 3)
        
        if date_range == "Custom":
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
        
        st.markdown("---")
        
        # Quick Stats
        if 'comparison_df' in st.session_state and st.session_state['comparison_df'] is not None:
            comparison_df = st.session_state['comparison_df']
            total_cost = comparison_df['Current_Cost'].sum()
            
            st.metric("💰 Current Month", f"${total_cost:,.0f}")
            
            if len(comparison_df) > 0:
                change = comparison_df['Cost_Difference'].sum()
                st.metric("📈 Month Change", f"${change:,.0f}")
    
    # Main Content Area
    if page == "🏠 Executive Dashboard":
        show_executive_dashboard(analyzer, months_back, user)
    
    elif page == "💰 Cost Analysis":
        show_cost_analysis(analyzer, months_back, user, claude_key)
    
    elif page == "🌍 Multi-Account View":
        show_multi_account_view(analyzer, months_back, user)
    
    elif page == "🏷️ Tag-Based Analysis":
        show_tag_analysis(analyzer, months_back, user)
    
    elif page == "📊 RI & Savings Plans":
        show_ri_savings_analysis(analyzer, months_back, user)
    
    elif page == "🚨 Anomaly Detection":
        show_anomaly_detection(analyzer, months_back, user)
    
    elif page == "💡 Optimization Insights":
        show_optimization_insights(analyzer, months_back, user, claude_key)
    
    elif page == "📈 Custom Reports":
        show_custom_reports(analyzer, user)
    
    elif page == "⚙️ Settings & Budgets":
        show_settings_budgets(user)


def show_executive_dashboard(analyzer, months_back, user):
    """Executive dashboard with key metrics"""
    st.header("🏠 Executive Dashboard")
    st.caption("High-level overview of AWS spending")
    
    with st.spinner("Loading dashboard..."):
        monthly_df, comparison_df = analyzer.get_monthly_comparison(months_back)
        
        if monthly_df is None or len(comparison_df) == 0:
            st.error("No data available")
            return
        
        regional_df = analyzer.get_regional_breakdown(months_back)
        
        st.session_state['monthly_df'] = monthly_df
        st.session_state['comparison_df'] = comparison_df
        st.session_state['regional_df'] = regional_df
    
    # Key Metrics
    total_current = comparison_df['Current_Cost'].sum()
    total_previous = comparison_df['Previous_Cost'].sum()
    total_change = comparison_df['Cost_Difference'].sum()
    change_pct = (total_change / total_previous * 100) if total_previous > 0 else 0
    
    # Burn rate
    analytics = AdvancedAnalytics()
    burn_rate = analytics.calculate_burn_rate(monthly_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Current Month Cost",
            f"${total_current:,.2f}",
            f"{change_pct:+.1f}%"
        )
    
    with col2:
        st.metric(
            "📊 Previous Month",
            f"${total_previous:,.2f}"
        )
    
    with col3:
        if burn_rate:
            st.metric(
                "🔥 Daily Burn Rate",
                f"${burn_rate['daily_burn_rate']:,.2f}/day"
            )
    
    with col4:
        st.metric(
            "🎯 Services Tracked",
            len(comparison_df)
        )
    
    st.markdown("---")
    
    # Budget Alerts
    budgets = st.session_state.get('budgets', {'monthly_total': 10000})
    alerts = BudgetManager.check_budget_alerts(comparison_df, budgets)
    
    if alerts:
        st.subheader("🚨 Budget Alerts")
        for alert in alerts:
            if alert['severity'] == 'high':
                st.error(alert['message'])
            else:
                st.warning(alert['message'])
    
    st.markdown("---")
    
    # Charts
    st.subheader("📊 Key Visualizations")
    
    fig_trend, fig_treemap, fig_waterfall, fig_heatmap = create_enterprise_dashboard_charts(
        monthly_df, comparison_df, regional_df
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_trend, use_container_width=True)
    with col2:
        st.plotly_chart(fig_treemap, use_container_width=True)
    
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Top Cost Drivers
    st.subheader("💼 Top Cost Drivers")
    top_10 = comparison_df.nlargest(10, 'Current_Cost')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(
            top_10[['Service', 'Current_Cost', 'Cost_Difference', 'Cost_Change_%']].style.format({
                'Current_Cost': '${:,.2f}',
                'Cost_Difference': '${:,.2f}',
                'Cost_Change_%': '{:.1f}%'
            }).background_gradient(subset=['Current_Cost'], cmap='Blues'),
            use_container_width=True,
            height=400
        )
    
    with col2:
        # Optimization opportunities
        opportunities = analytics.identify_optimization_opportunities(comparison_df)
        st.markdown("**🎯 Quick Wins:**")
        for opp in opportunities[:3]:
            st.info(f"**{opp['service']}**: {opp['description']}")


def show_cost_analysis(analyzer, months_back, user, claude_key):
    """Detailed cost analysis page"""
    st.header("💰 Detailed Cost Analysis")
    
    if 'monthly_df' not in st.session_state:
        monthly_df, comparison_df = analyzer.get_monthly_comparison(months_back)
        st.session_state['monthly_df'] = monthly_df
        st.session_state['comparison_df'] = comparison_df
    else:
        monthly_df = st.session_state['monthly_df']
        comparison_df = st.session_state['comparison_df']
    
    # Service filter
    all_services = sorted(comparison_df['Service'].unique())
    selected_services = st.multiselect(
        "Filter by Services",
        all_services,
        default=all_services[:10]
    )
    
    if selected_services:
        filtered_df = comparison_df[comparison_df['Service'].isin(selected_services)]
    else:
        filtered_df = comparison_df
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 All Services",
        "📈 Increases",
        "📉 Decreases",
        "🔍 Search"
    ])
    
    with tab1:
        st.dataframe(
            filtered_df.sort_values('Current_Cost', ascending=False).style.format({
                'Current_Cost': '${:,.2f}',
                'Previous_Cost': '${:,.2f}',
                'Cost_Difference': '${:,.2f}',
                'Cost_Change_%': '{:.1f}%'
            }),
            use_container_width=True,
            height=500
        )
    
    with tab2:
        increases = filtered_df[filtered_df['Cost_Difference'] > 0].nlargest(20, 'Cost_Difference')
        st.dataframe(
            increases[['Service', 'Current_Cost', 'Cost_Difference', 'Cost_Change_%']].style.format({
                'Current_Cost': '${:,.2f}',
                'Cost_Difference': '${:,.2f}',
                'Cost_Change_%': '{:.1f}%'
            }).background_gradient(subset=['Cost_Difference'], cmap='Reds'),
            use_container_width=True,
            height=500
        )
    
    with tab3:
        decreases = filtered_df[filtered_df['Cost_Difference'] < 0].nsmallest(20, 'Cost_Difference')
        st.dataframe(
            decreases[['Service', 'Current_Cost', 'Cost_Difference', 'Cost_Change_%']].style.format({
                'Current_Cost': '${:,.2f}',
                'Cost_Difference': '${:,.2f}',
                'Cost_Change_%': '{:.1f}%'
            }).background_gradient(subset=['Cost_Difference'], cmap='Greens_r'),
            use_container_width=True,
            height=500
        )
    
    with tab4:
        search_term = st.text_input("Search for service")
        if search_term:
            search_results = filtered_df[filtered_df['Service'].str.contains(search_term, case=False, na=False)]
            st.dataframe(search_results, use_container_width=True)
    
    # AI Insights (if user has permission)
    if AuthManager.check_permission(user, 'view'):
        st.markdown("---")
        st.subheader("🤖 AI Cost Analysis")
        
        if st.button("✨ Generate AI Insights", type="primary"):
            with st.spinner("Analyzing..."):
                from anthropic import Anthropic
                client = Anthropic(api_key=claude_key)
                
                summary = {
                    'total': comparison_df['Current_Cost'].sum(),
                    'change': comparison_df['Cost_Difference'].sum(),
                    'top_increases': comparison_df.nlargest(5, 'Cost_Difference')[
                        ['Service', 'Cost_Difference']
                    ].to_dict('records')
                }
                
                prompt = f"""Analyze AWS costs:
Total: ${summary['total']:.2f}
Change: ${summary['change']:.2f}
Top increases: {json.dumps(summary['top_increases'], indent=2)}

Provide executive summary and 5 actionable recommendations."""

                message = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown(message.content[0].text)
                st.markdown('</div>', unsafe_allow_html=True)


def show_multi_account_view(analyzer, months_back, user):
    """Multi-account cost view with scalability for 100+ accounts"""
    st.header("🌍 Multi-Account Cost Analysis")
    st.caption("Consolidated view across AWS accounts")
    
    # Get linked accounts
    accounts = analyzer.get_linked_accounts()
    
    # Get account costs
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=months_back * 30)
    
    with st.spinner("Loading account costs..."):
        account_df = analyzer.get_cost_by_account(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    if account_df.empty:
        st.info("ℹ️ Single account mode - enable AWS Organizations for multi-account view")
        return
    
    # Process data
    account_df['Period'] = pd.to_datetime(account_df['Period'])
    current_period = account_df['Period'].max()
    current_accounts = account_df[account_df['Period'] == current_period].sort_values('Cost', ascending=False)
    
    # Create account mapping with names
    account_names = {}
    if accounts:
        for acc in accounts:
            account_names[acc['Id']] = acc['Name']
    
    # Add account names to dataframe
    if account_names:
        current_accounts['AccountName'] = current_accounts['AccountId'].map(
            lambda x: account_names.get(x, x)
        )
        account_df['AccountName'] = account_df['AccountId'].map(
            lambda x: account_names.get(x, x)
        )
    else:
        current_accounts['AccountName'] = current_accounts['AccountId']
        account_df['AccountName'] = account_df['AccountId']
    
    # =========================
    # CONSOLIDATED VIEW (ALWAYS SHOWN)
    # =========================
    st.subheader("📊 Consolidated View - All Accounts")
    
    total_accounts = len(current_accounts)
    total_cost = current_accounts['Cost'].sum()
    avg_cost = current_accounts['Cost'].mean()
    max_account = current_accounts.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Accounts", total_accounts)
    with col2:
        st.metric("Total Cost", f"${total_cost:,.2f}")
    with col3:
        st.metric("Average Cost/Account", f"${avg_cost:,.2f}")
    with col4:
        st.metric("Highest Cost", f"${max_account['Cost']:,.2f}")
    
    # Cost distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 accounts pie chart
        top_10_accounts = current_accounts.head(10)
        other_cost = total_cost - top_10_accounts['Cost'].sum()
        
        if other_cost > 0:
            # Add "Others" category
            display_data = pd.concat([
                top_10_accounts[['AccountName', 'Cost']],
                pd.DataFrame([{'AccountName': 'Others', 'Cost': other_cost}])
            ])
        else:
            display_data = top_10_accounts[['AccountName', 'Cost']]
        
        fig_pie = px.pie(
            display_data,
            values='Cost',
            names='AccountName',
            title=f'Cost Distribution (Top 10 of {total_accounts} accounts)',
            hole=0.3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Top accounts table
        st.markdown("**🏆 Top 10 Cost Accounts**")
        st.dataframe(
            top_10_accounts[['AccountName', 'AccountId', 'Cost']].style.format({
                'Cost': '${:,.2f}'
            }),
            use_container_width=True,
            height=400
        )
    
    st.markdown("---")
    
    # =========================
    # ACCOUNT SELECTOR
    # =========================
    st.subheader("🔍 Detailed Account Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Create account options
        account_options = ["📊 All Accounts (Overview)"] + [
            f"{row['AccountName']} ({row['AccountId']}) - ${row['Cost']:,.2f}"
            for _, row in current_accounts.iterrows()
        ]
        
        selected_account = st.selectbox(
            "Select Account",
            account_options,
            help="Choose an account to view detailed analysis"
        )
    
    with col2:
        # Search filter
        search_term = st.text_input("🔎 Search Account", placeholder="Name or ID")
    
    with col3:
        # Sort options
        sort_by = st.selectbox("Sort By", ["Cost (High to Low)", "Cost (Low to High)", "Name (A-Z)", "ID"])
    
    # =========================
    # FILTER & SORT ACCOUNTS
    # =========================
    filtered_accounts = current_accounts.copy()
    
    # Apply search filter
    if search_term:
        filtered_accounts = filtered_accounts[
            filtered_accounts['AccountName'].str.contains(search_term, case=False, na=False) |
            filtered_accounts['AccountId'].str.contains(search_term, case=False, na=False)
        ]
    
    # Apply sorting
    if sort_by == "Cost (High to Low)":
        filtered_accounts = filtered_accounts.sort_values('Cost', ascending=False)
    elif sort_by == "Cost (Low to High)":
        filtered_accounts = filtered_accounts.sort_values('Cost', ascending=True)
    elif sort_by == "Name (A-Z)":
        filtered_accounts = filtered_accounts.sort_values('AccountName')
    elif sort_by == "ID":
        filtered_accounts = filtered_accounts.sort_values('AccountId')
    
    st.markdown("---")
    
    # =========================
    # SHOW SELECTED VIEW
    # =========================
    
    if selected_account == "📊 All Accounts (Overview)":
        # SHOW ALL ACCOUNTS VIEW
        st.subheader("📋 All Accounts Overview")
        
        tabs = st.tabs(["📊 Table View", "📈 Charts", "📉 Trends"])
        
        with tabs[0]:
            # Paginated table view
            st.markdown(f"**Showing {len(filtered_accounts)} of {total_accounts} accounts**")
            
            # Add cost categories
            filtered_accounts['Category'] = pd.cut(
                filtered_accounts['Cost'],
                bins=[0, 100, 1000, 10000, float('inf')],
                labels=['💚 Low (<$100)', '💛 Medium ($100-$1K)', '🟠 High ($1K-$10K)', '🔴 Very High (>$10K)']
            )
            
            st.dataframe(
                filtered_accounts[['Category', 'AccountName', 'AccountId', 'Cost']].style.format({
                    'Cost': '${:,.2f}'
                }),
                use_container_width=True,
                height=600
            )
            
            # Export button
            csv = filtered_accounts[['AccountName', 'AccountId', 'Cost']].to_csv(index=False)
            st.download_button(
                "📥 Download Account List (CSV)",
                csv,
                f"all_accounts_{datetime.now().strftime('%Y%m%d')}.csv",
                use_container_width=True
            )
        
        with tabs[1]:
            # Bar chart for all accounts (top 20)
            st.markdown("**Top 20 Accounts by Cost**")
            top_20 = filtered_accounts.head(20)
            
            fig_bar = px.bar(
                top_20,
                y='AccountName',
                x='Cost',
                orientation='h',
                title='Top 20 Accounts',
                color='Cost',
                color_continuous_scale='Reds',
                text='Cost'
            )
            fig_bar.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig_bar.update_layout(height=max(400, len(top_20) * 30))
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with tabs[2]:
            # Trend for top 10 accounts
            st.markdown("**Cost Trends - Top 10 Accounts**")
            top_10_ids = current_accounts.head(10)['AccountId'].tolist()
            trend_data = account_df[account_df['AccountId'].isin(top_10_ids)]
            
            fig_trend = px.line(
                trend_data,
                x='Period',
                y='Cost',
                color='AccountName',
                title='Cost Trends (Top 10 Accounts)',
                markers=True
            )
            fig_trend.update_layout(height=500)
            st.plotly_chart(fig_trend, use_container_width=True)
    
    else:
        # SHOW SINGLE ACCOUNT DETAILED VIEW
        # Extract account ID from selection
        account_id = selected_account.split('(')[1].split(')')[0]
        account_data = current_accounts[current_accounts['AccountId'] == account_id].iloc[0]
        account_name = account_data['AccountName']
        
        st.subheader(f"📊 Detailed Analysis: {account_name}")
        st.caption(f"Account ID: {account_id}")
        
        # Account metrics
        account_cost = account_data['Cost']
        account_pct = (account_cost / total_cost * 100)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Month Cost", f"${account_cost:,.2f}")
        with col2:
            st.metric("% of Total", f"{account_pct:.1f}%")
        with col3:
            rank = filtered_accounts[filtered_accounts['AccountId'] == account_id].index[0] + 1
            st.metric("Cost Rank", f"#{rank} of {total_accounts}")
        with col4:
            # Calculate vs average
            vs_avg = account_cost - avg_cost
            st.metric("vs Avg", f"${vs_avg:,.2f}", f"{(vs_avg/avg_cost*100):+.1f}%")
        
        st.markdown("---")
        
        # Account trend
        account_history = account_df[account_df['AccountId'] == account_id].sort_values('Period')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Cost Trend**")
            fig_account_trend = px.line(
                account_history,
                x='Period',
                y='Cost',
                markers=True,
                title=f'{account_name} - Cost History'
            )
            fig_account_trend.update_traces(line_color='#FF9900', line_width=3)
            st.plotly_chart(fig_account_trend, use_container_width=True)
        
        with col2:
            st.markdown("**📊 Historical Data**")
            st.dataframe(
                account_history[['Period', 'Cost']].style.format({
                    'Cost': '${:,.2f}',
                    'Period': lambda x: x.strftime('%Y-%m')
                }),
                use_container_width=True,
                height=400
            )
        
        # Month-over-month change
        if len(account_history) >= 2:
            current_cost = account_history.iloc[-1]['Cost']
            previous_cost = account_history.iloc[-2]['Cost']
            mom_change = current_cost - previous_cost
            mom_pct = (mom_change / previous_cost * 100) if previous_cost > 0 else 0
            
            if mom_pct > 10:
                st.warning(f"⚠️ **Alert:** This account's cost increased by **${mom_change:,.2f} ({mom_pct:+.1f}%)** compared to last month")
            elif mom_pct < -10:
                st.success(f"✅ **Good news:** This account's cost decreased by **${abs(mom_change):,.2f} ({mom_pct:.1f}%)** compared to last month")
            else:
                st.info(f"📊 Month-over-month change: **${mom_change:,.2f} ({mom_pct:+.1f}%)**")
        
        # Service breakdown for this account (if available)
        st.markdown("---")
        st.markdown("**💡 Recommendation:**")
        st.info(f"""
        **For account {account_name}:**
        - Current spend: ${account_cost:,.2f}/month (${account_cost/30:.2f}/day)
        - Represents {account_pct:.1f}% of total organization cost
        - Consider reviewing top services in this account for optimization opportunities
        """)
    
    # Account grouping (optional)
    with st.expander("🏷️ Group Accounts by Cost Range"):
        st.markdown("**Account Distribution by Cost Range:**")
        
        cost_ranges = pd.cut(
            current_accounts['Cost'],
            bins=[0, 100, 1000, 10000, float('inf')],
            labels=['< $100', '$100 - $1,000', '$1,000 - $10,000', '> $10,000']
        )
        
        range_summary = current_accounts.groupby(cost_ranges, observed=False).agg({
            'AccountId': 'count',
            'Cost': 'sum'
        }).rename(columns={'AccountId': 'Count', 'Cost': 'Total Cost'})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(
                range_summary.style.format({
                    'Total Cost': '${:,.2f}'
                }),
                use_container_width=True
            )
        
        with col2:
            fig_ranges = px.pie(
                values=range_summary['Count'],
                names=range_summary.index,
                title='Account Distribution'
            )
            st.plotly_chart(fig_ranges, use_container_width=True)


def show_tag_analysis(analyzer, months_back, user):
    """Tag-based cost analysis"""
    st.header("🏷️ Tag-Based Cost Analysis")
    st.caption("Analyze costs by custom tags (Environment, Team, Project, etc.)")
    
    tag_key = st.selectbox(
        "Select Tag Key",
        ["Environment", "Team", "Project", "CostCenter", "Application", "Owner"]
    )
    
    if st.button("Analyze by Tag", type="primary"):
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=months_back * 30)
        
        with st.spinner(f"Fetching costs by {tag_key}..."):
            tag_df = analyzer.get_cost_by_tags(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                tag_key
            )
        
        if not tag_df.empty:
            tag_df['Period'] = pd.to_datetime(tag_df['Period'])
            current_period = tag_df['Period'].max()
            current_tags = tag_df[tag_df['Period'] == current_period].sort_values('Cost', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(
                    current_tags,
                    values='Cost',
                    names='TagValue',
                    title=f'Cost Distribution by {tag_key}'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.dataframe(
                    current_tags[['TagValue', 'Cost']].style.format({'Cost': '${:,.2f}'}),
                    use_container_width=True,
                    height=400
                )
            
            # Trend
            fig_trend = px.line(
                tag_df,
                x='Period',
                y='Cost',
                color='TagValue',
                title=f'Cost Trend by {tag_key}'
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning(f"No cost data found for tag: {tag_key}")
            st.info("Make sure Cost Allocation Tags are enabled in AWS Billing settings")


def show_ri_savings_analysis(analyzer, months_back, user):
    """RI and Savings Plans analysis"""
    st.header("📊 Reserved Instances & Savings Plans Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔒 Reserved Instance Coverage")
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=months_back * 30)
        
        ri_df = analyzer.get_ri_coverage(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if not ri_df.empty:
            ri_df['Period'] = pd.to_datetime(ri_df['Period'])
            
            avg_coverage = ri_df['CoverageHours'].mean()
            
            st.metric("Average RI Coverage", f"{avg_coverage:.1f}%")
            
            fig_ri = px.line(
                ri_df,
                x='Period',
                y='CoverageHours',
                title='RI Coverage Trend'
            )
            st.plotly_chart(fig_ri, use_container_width=True)
            
            if avg_coverage < 70:
                st.warning("⚠️ Consider purchasing more RIs to improve coverage")
        else:
            st.info("No RI coverage data available")
    
    with col2:
        st.subheader("💰 Savings Plans Coverage")
        
        sp_df = analyzer.get_savings_plans_coverage(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if not sp_df.empty:
            sp_df['Period'] = pd.to_datetime(sp_df['Period'])
            
            avg_sp_coverage = sp_df['CoveragePercentage'].mean()
            
            st.metric("Average SP Coverage", f"{avg_sp_coverage:.1f}%")
            
            fig_sp = px.line(
                sp_df,
                x='Period',
                y='CoveragePercentage',
                title='Savings Plans Coverage Trend'
            )
            st.plotly_chart(fig_sp, use_container_width=True)
            
            total_savings = sp_df['SpendCoveredBySavingsPlans'].sum()
            st.success(f"💰 Total covered by Savings Plans: ${total_savings:,.2f}")
        else:
            st.info("No Savings Plans data available")
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    recommendations = []
    
    if not ri_df.empty and ri_df['CoverageHours'].mean() < 70:
        recommendations.append("Consider purchasing Reserved Instances to reduce costs by up to 75%")
    
    if not sp_df.empty and sp_df['CoveragePercentage'].mean() < 60:
        recommendations.append("Savings Plans could reduce costs by up to 72% - review eligibility")
    
    if recommendations:
        for rec in recommendations:
            st.info(f"💡 {rec}")
    else:
        st.success("✅ Good coverage! Continue monitoring for optimization opportunities")


def show_anomaly_detection(analyzer, months_back, user):
    """Cost anomaly detection"""
    st.header("🚨 Cost Anomaly Detection")
    st.caption("AI-powered detection of unusual cost patterns")
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)  # Last 90 days
    
    with st.spinner("Detecting anomalies..."):
        anomalies = analyzer.get_cost_anomalies(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    
    if anomalies:
        st.error(f"🚨 Found {len(anomalies)} cost anomalies")
        
        for i, anomaly in enumerate(anomalies[:10], 1):
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                st.write(f"**{anomaly['Date']}**")
            with col2:
                st.write(f"Service: {anomaly['Service']}")
            with col3:
                st.write(f"Impact: ${anomaly['Impact']:,.2f}")
            with col4:
                score = anomaly['Score']
                if score > 80:
                    st.error(f"⚠️ {score:.0f}")
                else:
                    st.warning(f"⚠️ {score:.0f}")
        
        # Chart
        anomaly_df = pd.DataFrame(anomalies)
        fig = px.scatter(
            anomaly_df,
            x='Date',
            y='Impact',
            size='Score',
            color='Service',
            title='Cost Anomalies Timeline',
            hover_data=['Score']
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No significant cost anomalies detected")
        st.info("💡 Enable AWS Cost Anomaly Detection in the AWS Console for enhanced monitoring")


def show_optimization_insights(analyzer, months_back, user, claude_key):
    """AI-powered optimization insights"""
    st.header("💡 Cost Optimization Insights")
    
    if 'comparison_df' not in st.session_state:
        monthly_df, comparison_df = analyzer.get_monthly_comparison(months_back)
        st.session_state['comparison_df'] = comparison_df
    else:
        comparison_df = st.session_state['comparison_df']
    
    analytics = AdvancedAnalytics()
    opportunities = analytics.identify_optimization_opportunities(comparison_df)
    
    st.subheader("🎯 Identified Opportunities")
    
    for opp in opportunities:
        if opp['type'] == 'rightsizing':
            st.info(f"""
            **💰 Rightsizing Opportunity: {opp['service']}**
            - Current Cost: ${opp['cost']:,.2f}
            - Action: {opp['description']}
            """)
        elif opp['type'] == 'growth_alert':
            st.warning(f"""
            **📈 Rapid Growth Alert: {opp['service']}**
            - Growth Rate: {opp['growth']:.1f}%
            - Action: {opp['description']}
            """)
    
    st.markdown("---")
    
    # AI-Powered Deep Dive
    if AuthManager.check_permission(user, 'view'):
        st.subheader("🤖 AI-Powered Optimization Analysis")
        
        if st.button("🚀 Generate Comprehensive Optimization Plan", type="primary"):
            with st.spinner("Claude is analyzing your infrastructure..."):
                from anthropic import Anthropic
                client = Anthropic(api_key=claude_key)
                
                top_costs = comparison_df.nlargest(10, 'Current_Cost')
                
                prompt = f"""As an AWS cost optimization expert, analyze:

**Top 10 Services:**
{top_costs[['Service', 'Current_Cost', 'Cost_Change_%']].to_dict('records')}

Provide:
1. **Executive Summary** (2-3 sentences)
2. **Top 5 Optimization Opportunities** (specific, actionable)
3. **Quick Wins** (can be done immediately)
4. **Strategic Recommendations** (long-term)
5. **Estimated Savings Potential**
6. **Implementation Priority** (High/Medium/Low for each)

Format with clear headers and bullet points."""

                message = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=2500,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown(message.content[0].text)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.session_state['optimization_plan'] = message.content[0].text
        
        # Export optimization plan
        if 'optimization_plan' in st.session_state and AuthManager.check_permission(user, 'export'):
            st.download_button(
                "📥 Download Optimization Plan",
                st.session_state['optimization_plan'],
                f"aws_optimization_plan_{datetime.now().strftime('%Y%m%d')}.txt",
                use_container_width=True
            )


def show_custom_reports(analyzer, user):
    """Custom report builder"""
    st.header("📈 Custom Report Builder")
    
    if not AuthManager.check_permission(user, 'export'):
        st.warning("⚠️ You don't have permission to create custom reports")
        return
    
    st.subheader("📊 Build Your Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_name = st.text_input("Report Name", "Monthly Cost Report")
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Analysis", "Service Breakdown", "Regional Analysis"]
        )
        
        include_charts = st.checkbox("Include Charts", value=True)
        include_recommendations = st.checkbox("Include AI Recommendations", value=True)
    
    with col2:
        date_range = st.selectbox("Date Range", ["Last month", "Last 3 months", "Last 6 months", "Last 12 months"])
        format_type = st.selectbox("Export Format", ["PDF", "Excel", "CSV", "JSON"])
        
        schedule = st.selectbox("Schedule", ["One-time", "Daily", "Weekly", "Monthly"])
    
    if st.button("📊 Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            # Simulate report generation
            time.sleep(2)
            
            st.success(f"✅ Report '{report_name}' generated successfully!")
            
            # Download button
            st.download_button(
                "📥 Download Report",
                "Sample report content...",  # In production, generate actual report
                f"{report_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.{format_type.lower()}",
                use_container_width=True
            )


def show_settings_budgets(user):
    """Settings and budget management"""
    st.header("⚙️ Settings & Budget Management")
    
    if not AuthManager.check_permission(user, 'budget'):
        st.warning("⚠️ You don't have permission to manage budgets")
        return
    
    tab1, tab2, tab3 = st.tabs(["💰 Budgets", "🔔 Alerts", "👥 User Management"])
    
    with tab1:
        st.subheader("💰 Budget Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_budget = st.number_input(
                "Monthly Total Budget ($)",
                min_value=0,
                value=st.session_state.get('budgets', {}).get('monthly_total', 10000),
                step=1000
            )
            
            if st.button("Save Monthly Budget"):
                if 'budgets' not in st.session_state:
                    st.session_state['budgets'] = {}
                st.session_state['budgets']['monthly_total'] = monthly_budget
                st.success("✅ Budget saved!")
        
        with col2:
            st.info(f"""
            **Current Budget Status:**
            - Monthly Budget: ${monthly_budget:,.2f}
            - Current Spend: ${st.session_state.get('comparison_df', pd.DataFrame()).get('Current_Cost', pd.Series()).sum():,.2f}
            - Remaining: ${monthly_budget - st.session_state.get('comparison_df', pd.DataFrame()).get('Current_Cost', pd.Series()).sum():,.2f}
            """)
        
        st.markdown("---")
        st.subheader("🎯 Service-Level Budgets")
        
        service_budget_name = st.text_input("Service Name", "Amazon EC2")
        service_budget_amount = st.number_input("Budget Amount ($)", min_value=0, value=1000, step=100)
        
        if st.button("Add Service Budget"):
            if 'budgets' not in st.session_state:
                st.session_state['budgets'] = {'services': {}}
            if 'services' not in st.session_state['budgets']:
                st.session_state['budgets']['services'] = {}
            
            st.session_state['budgets']['services'][service_budget_name] = service_budget_amount
            st.success(f"✅ Budget for {service_budget_name} set to ${service_budget_amount}")
    
    with tab2:
        st.subheader("🔔 Alert Configuration")
        
        alert_email = st.text_input("Alert Email", "finance@company.com")
        
        alert_thresholds = st.multiselect(
            "Alert When Budget Reaches",
            ["50%", "75%", "90%", "100%"],
            default=["75%", "90%", "100%"]
        )
        
        anomaly_alerts = st.checkbox("Enable Anomaly Detection Alerts", value=True)
        
        if st.button("Save Alert Settings"):
            st.success("✅ Alert settings saved!")
    
    with tab3:
        if user['role'] == 'admin':
            st.subheader("👥 User Management")
            
            st.dataframe(
                pd.DataFrame([
                    {"Username": "admin", "Role": "Admin", "Status": "Active"},
                    {"Username": "finance", "Role": "Finance", "Status": "Active"},
                    {"Username": "viewer", "Role": "Viewer", "Status": "Active"}
                ]),
                use_container_width=True
            )
            
            with st.expander("➕ Add New User"):
                new_username = st.text_input("Username")
                new_role = st.selectbox("Role", ["Admin", "Finance", "Viewer"])
                new_password = st.text_input("Password", type="password")
                
                if st.button("Create User"):
                    st.success(f"✅ User {new_username} created!")
        else:
            st.warning("⚠️ Only administrators can manage users")


if __name__ == "__main__":
    main()