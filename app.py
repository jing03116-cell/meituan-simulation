"""
AI用户仿真平台 - Streamlit Cloud版本
适用于互联网商业分析的智能补贴推演沙盘
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 页面配置（Streamlit Cloud必需）
# ============================================
st.set_page_config(
    page_title="AI用户仿真平台 | 智能补贴推演",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 全局样式
# ============================================
st.markdown("""
<style>
    /* 专业级样式 */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #6b7280;
        font-size: 0.9rem;
        text-transform: uppercase;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 标题
# ============================================
st.markdown("""
<div class="main-header">
    <h1 style="margin:0">🎯 AI用户仿真平台</h1>
    <p style="margin-top:0.5rem; opacity:0.9; font-size:1.1rem">
        智能补贴推演沙盘 · 因果推断 · 策略优化
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================
# 侧边栏配置
# ============================================
with st.sidebar:
    st.markdown("## 🎮 仿真控制台")
    
    st.markdown("### 📊 基础配置")
    
    # 场景选择
    scenario = st.selectbox(
        "业务场景",
        ["周末下午茶", "暴雨晚高峰", "节假日出行", "日常通勤"],
        help="不同场景影响用户打开率和转化率"
    )
    
    # 城市规模
    city_scale = st.select_slider(
        "城市规模（日活用户）",
        options=["10万", "50万", "100万", "200万", "500万"],
        value="100万"
    )
    
    # 转换为数字
    scale_map = {"10万": 100000, "50万": 500000, "100万": 1000000, 
                 "200万": 2000000, "500万": 5000000}
    dau = scale_map[city_scale]
    
    st.markdown("### 💰 补贴策略")
    
    # 券面额
    coupon_amount = st.slider(
        "券面额（元）",
        min_value=3,
        max_value=20,
        value=8,
        step=1
    )
    
    # 使用门槛
    threshold = st.slider(
        "使用门槛（元）",
        min_value=20,
        max_value=100,
        value=40,
        step=5
    )
    
    # 券类型
    coupon_type = st.radio(
        "券类型",
        ["免费券", "付费券包（神券包）"],
        help="付费券包：用户购买券包，平台净成本更低"
    )
    
    st.markdown("### 🎯 优化目标")
    
    target_roi = st.slider(
        "目标ROI",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5
    )
    
    st.markdown("---")
    
    # 运行按钮
    run_simulation = st.button(
        "🚀 运行仿真",
        type="primary",
        use_container_width=True
    )

# ============================================
# 仿真引擎（替换为读取大宽表数据）
# ============================================
@st.cache_data
def load_real_user_data():
    """从真实的宽表中读取数据，并映射为仿真逻辑所需的特征"""
    try:
        df = pd.read_csv("大宽表.csv")
    except FileNotFoundError:
        st.error("⚠️ 未在当前目录找到【大宽表.csv】文件，请确保文件已上传！")
        st.stop()

    # 映射真实列名为现有仿真逻辑使用的列名
    mapping = {
        "画像名称": "persona",
        "平均客单价": "avg_order_value",
        "补贴覆盖率": "price_sensitivity",      # 借用补贴覆盖率映射为价格敏感度
        "用券率": "coupon_sensitivity",         # 借用历史用券率映射为券敏感度
        "动态_点击_至_加购率": "base_conversion_rate" # 借用历史加购率映射为基础转化率
    }
    
    # 保留需要的列并重命名
    if "user_id" not in df.columns:
        df["user_id"] = ["U" + str(i).zfill(6) for i in range(len(df))]
        
    df = df.rename(columns=mapping)
    
    # 处理空值与边界约束，保障模型稳定
    for col in ['avg_order_value', 'price_sensitivity', 'coupon_sensitivity', 'base_conversion_rate']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
            
    # 确保客单价不低于仿真设定的15元底线
    if 'avg_order_value' in df.columns:
        df['avg_order_value'] = df['avg_order_value'].clip(lower=15)
        
    return df

def simulate_conversion(user_row, scenario, coupon_info):
    """模拟用户转化决策"""
    
    # 场景因子
    scenario_factors = {
        "周末下午茶": {"open_rate": 0.45, "pay_rate": 0.35},
        "暴雨晚高峰": {"open_rate": 0.65, "pay_rate": 0.25},
        "节假日出行": {"open_rate": 0.55, "pay_rate": 0.40},
        "日常通勤": {"open_rate": 0.50, "pay_rate": 0.30}
    }
    
    factors = scenario_factors[scenario]
    
    # 基础转化概率
    base_conv = user_row['base_conversion_rate']
    
    # 券的吸引力
    coupon_appeal = min(
        coupon_info['amount'] / user_row['avg_order_value'] * user_row['coupon_sensitivity'],
        1.0
    )
    
    # 价格敏感度影响
    if user_row['avg_order_value'] >= coupon_info['threshold']:
        threshold_met = 1.2  # 达到门槛，转化率提升
    else:
        threshold_met = 0.3  # 未达到门槛，大幅降低
    
    # 计算最终转化概率
    final_conv = (
        base_conv * 0.3 +
        factors['open_rate'] * 0.2 +
        coupon_appeal * 0.3 +
        (1 - user_row['price_sensitivity']) * 0.2
    ) * threshold_met
    
    # 付费券包的额外加成 (使用外部全局变量)
    if coupon_type == "付费券包（神券包）":
        final_conv *= 1.15  # 沉没成本效应
    
    return np.clip(final_conv, 0.05, 0.95)

def run_simulation_model(df, scenario, coupon_amount, threshold, target_roi):
    """运行仿真模型"""
    
    # 场景映射
    scenario_map = {
        "周末下午茶": 0.8,
        "暴雨晚高峰": 1.2,
        "节假日出行": 1.0,
        "日常通勤": 0.9
    }
    
    # 成本计算 (使用外部全局变量)
    if coupon_type == "付费券包（神券包）":
        # 神券包：用户购买券包，平台实际成本 = 券面额 - 购买费
        platform_cost = coupon_amount * 0.7  # 假设用户购买成本占30%
    else:
        platform_cost = coupon_amount
    
    # 逐用户仿真
    results = []
    for _, user in df.iterrows():
        conv_prob = simulate_conversion(user, scenario, 
                                       {"amount": coupon_amount, "threshold": threshold})
        
        # 是否转化
        converted = np.random.random() < conv_prob
        
        if converted:
            gtv = user['avg_order_value']
            cost = platform_cost
        else:
            gtv = 0
            cost = 0
        
        results.append({
            "user_id": user['user_id'],
            "persona": user['persona'],
            "converted": converted,
            "gtv": gtv,
            "cost": cost,
            "conv_prob": conv_prob
        })
    
    results_df = pd.DataFrame(results)
    
    # 聚合统计
    total_gtv = results_df['gtv'].sum()
    total_cost = results_df['cost'].sum()
    roi = total_gtv / total_cost if total_cost > 0 else float('inf')
    conversion_rate = results_df['converted'].mean()
    
    # 分群统计
    segment_stats = results_df.groupby('persona').agg({
        'converted': 'mean',
        'gtv': 'sum',
        'cost': 'sum'
    }).reset_index()
    segment_stats['roi'] = segment_stats['gtv'] / segment_stats['cost'].replace(0, np.nan)
    
    return {
        "total_gtv": total_gtv,
        "total_cost": total_cost,
        "roi": roi,
        "conversion_rate": conversion_rate,
        "avg_order_value": results_df[results_df['converted']]['gtv'].mean() if len(results_df[results_df['converted']]) > 0 else 0,
        "segment_stats": segment_stats,
        "user_results": results_df,
        "meets_target": roi >= target_roi
    }

# ============================================
# 加载真实数据 (替代了原有的 mock 数据生成)
# ============================================
df_users = load_real_user_data()

# ============================================
# 运行仿真
# ============================================
if run_simulation:
    with st.spinner("🔄 正在运行仿真模型..."):
        # 放大到城市规模
        scale_factor = dau / len(df_users)
        
        results = run_simulation_model(
            df_users, scenario, coupon_amount, threshold, target_roi
        )
        
        # 放大结果
        results['total_gtv_scaled'] = results['total_gtv'] * scale_factor
        results['total_cost_scaled'] = results['total_cost'] * scale_factor
        
        st.session_state['results'] = results
        st.session_state['simulation_run'] = True

# ============================================
# 显示结果
# ============================================
if 'simulation_run' in st.session_state and st.session_state['simulation_run']:
    results = st.session_state['results']
    
    # 关键指标卡片
    st.markdown("<div class='section-title'>📈 核心指标</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">总GTV</div>
            <div class="metric-value" style="color: #667eea;">
                ¥ {results['total_gtv_scaled']/10000:.1f}万
            </div>
            <div style="color: #10b981; font-size: 0.9rem;">
                转化率 {results['conversion_rate']:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cost_ratio = results['total_cost_scaled']/results['total_gtv_scaled']*100 if results['total_gtv_scaled'] > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">补贴成本</div>
            <div class="metric-value" style="color: #ef4444;">
                ¥ {results['total_cost_scaled']/10000:.1f}万
            </div>
            <div style="color: #6b7280; font-size: 0.9rem;">
                占比 {cost_ratio:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        roi_color = "#10b981" if results['meets_target'] else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">投资回报率</div>
            <div class="metric-value" style="color: {roi_color};">
                {results['roi']:.2f}x
            </div>
            <div style="color: #6b7280; font-size: 0.9rem;">
                目标 {target_roi:.1f}x
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        net_profit = results['total_gtv_scaled'] - results['total_cost_scaled']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">净收益</div>
            <div class="metric-value" style="color: #8b5cf6;">
                ¥ {net_profit/10000:.1f}万
            </div>
            <div style="color: #6b7280; font-size: 0.9rem;">
                客单价 ¥ {results['avg_order_value']:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 状态提示
    if results['meets_target']:
        st.success(f"✅ 当前策略达到ROI目标！ (实际 {results['roi']:.2f}x ≥ 目标 {target_roi:.1f}x)")
    else:
        st.warning(f"⚠️ 当前策略未达到ROI目标 (实际 {results['roi']:.2f}x < 目标 {target_roi:.1f}x)")
    
    # 详细分析
    st.markdown("<div class='section-title'>📊 详细分析</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 分群表现", "🎯 转化漏斗", "💰 成本效益"])
    
    with tab1:
        # 分群表现
        segment_df = results['segment_stats']
        
        fig_segment = go.Figure()
        
        fig_segment.add_trace(go.Bar(
            x=segment_df['persona'],
            y=segment_df['roi'],
            name='ROI',
            marker_color=['#667eea', '#ef4444', '#10b981', '#f59e0b'],
            text=[f"{x:.2f}x" if pd.notna(x) else "0.0x" for x in segment_df['roi']],
            textposition='outside'
        ))
        
        fig_segment.add_hline(y=target_roi, line_dash="dash", 
                              line_color="red", annotation_text=f"目标ROI: {target_roi}x")
        
        fig_segment.update_layout(
            title="各用户群体ROI对比",
            yaxis_title="ROI (x)",
            height=400
        )
        
        st.plotly_chart(fig_segment, use_container_width=True)
        
        # 数据表格
        st.dataframe(
            segment_df.style.format({
                'converted': '{:.1%}',
                'gtv': '¥ {:,.0f}',
                'cost': '¥ {:,.0f}',
                'roi': '{:.2f}x'
            }).background_gradient(subset=['roi'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    with tab2:
        # 转化漏斗
        total_users = len(df_users) * scale_factor
        opened = total_users * 0.6  # 假设打开率
        clicked = opened * 0.4
        converted = total_users * results['conversion_rate']
        
        fig_funnel = go.Figure(go.Funnel(
            y=['总用户', '打开APP', '浏览商家', '下单转化'],
            x=[total_users, opened, clicked, converted],
            textinfo="value+percent initial",
            marker={"color": ["#667eea", "#764ba2", "#f59e0b", "#10b981"]}
        ))
        
        fig_funnel.update_layout(title="用户转化漏斗", height=400)
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    with tab3:
        # 成本效益分析
        col1, col2 = st.columns(2)
        
        with col1:
            # ROI敏感性分析
            sensitivities = []
            for amount_mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
                test_amount = coupon_amount * amount_mult
                test_results = run_simulation_model(
                    df_users.sample(min(1000, len(df_users))), scenario, test_amount, threshold, target_roi
                )
                sensitivities.append({
                    '券面额': f"¥ {test_amount:.0f}",
                    'ROI': test_results['roi'],
                    '转化率': test_results['conversion_rate']
                })
            
            sens_df = pd.DataFrame(sensitivities)
            
            fig_sens = px.line(
                sens_df, x='券面额', y='ROI',
                title='券面额对ROI的影响',
                markers=True
            )
            fig_sens.add_hline(y=target_roi, line_dash="dash", line_color="red")
            st.plotly_chart(fig_sens, use_container_width=True)
        
        with col2:
            # 成本构成
            fig_cost = go.Figure(data=[
                go.Pie(
                    labels=['补贴成本', '净收益'],
                    values=[results['total_cost_scaled'], max(0, net_profit)], # 保护防跌破0报错
                    marker_colors=['#ef4444', '#10b981'],
                    hole=0.4
                )
            ])
            fig_cost.update_layout(title="成本收益构成")
            st.plotly_chart(fig_cost, use_container_width=True)

else:
    # 初始状态显示说明
    st.info("👈 请在左侧配置参数并点击「运行仿真」开始分析")
    
    # 展示示例图表
    st.markdown("<div class='section-title'>📊 平台能力预览</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 智能仿真
        - 多智能体行为模拟
        - 场景化决策引擎
        - 实时策略推演
        """)
    
    with col2:
        st.markdown("""
        ### 📈 因果分析
        - 增量效果评估
        - 反事实推理
        - 异质性分析
        """)
    
    with col3:
        st.markdown("""
        ### 🚀 策略优化
        - 贝叶斯优化
        - ROI最大化
        - 预算智能分配
        """)

# ============================================
# 页脚
# ============================================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>AI用户仿真平台 v2.0 | 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <p style="font-size: 0.8rem;">基于因果推断与多智能体强化学习 | Powered by Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
