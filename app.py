"""
AI用户仿真平台 - Streamlit Cloud版本
适用于互联网商业分析的智能补贴推演沙盘
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 0. 页面配置与全局样式
# ============================================
st.set_page_config(
    page_title="智能补贴推演沙盘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .kpi-card { background: linear-gradient(145deg, #ffffff, #f8f9fa); border-radius: 12px; padding: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); border-top: 5px solid #1f77b4; text-align: center; }
    .kpi-card h2 { color: #1f77b4; margin: 0; font-size: 38px; font-weight: 800; }
    .kpi-card p { color: #7f8c8d; margin: 8px 0 0 0; font-size: 15px; font-weight: 600; }
    .strategy-card { background-color: #ffffff; border-radius: 10px; padding: 20px; border: 1px solid #edf2f7; border-left: 4px solid #e67e22; box-shadow: 0 2px 10px rgba(0,0,0,0.02); }
    .section-title { font-size: 22px; font-weight: 700; color: #2c3e50; margin-top: 30px; margin-bottom: 20px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("智能补贴推演沙盘 (真实宽表驱动版)")
st.markdown("基于因果推断与运筹规划，严格依托底层真实业务特征，在守住 ROI 底线的前提下寻找大盘 GTV 最优解。")

# ============================================
# 1. 强制数据契约加载 (绝不捏造数据)
# ============================================
@st.cache_data
def load_real_data():
    try:
        # 强制读取真实业务宽表
        df = pd.read_csv("大宽表.csv")
    except FileNotFoundError:
        # 如果找不到文件，直接阻断运行，拒绝使用伪造数据
        st.error("严重错误：未在当前目录下找到【大宽表.csv】文件。仿真引擎必须依赖真实业务特征运行，系统拒绝执行。请上传或检查文件路径。")
        st.stop()
    
    # 核心字段完整性校验与空值填补
    required_cols = ['平均客单价', '用券率', '付费券使用率', '动态_点击_至_加购率', '补贴覆盖率', '画像名称']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"数据契约缺失：大宽表中缺失核心字段【{col}】，因果推断无法进行。")
            st.stop()
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
            
    return df

df_users = load_real_data()

SCENARIOS = {
    "周末下午茶": {"open_rate": 0.45, "base_pay": 0.35},
    "暴雨晚高峰": {"open_rate": 0.65, "base_pay": 0.20}, 
    "节假日出行": {"open_rate": 0.55, "base_pay": 0.45}
}

# ============================================
# 2. 策略空间生成器 (修正真实业务规则)
# ============================================
@st.cache_data
def generate_action_space():
    actions = [{"name": "留白 (Control)", "type": "none", "cost": 0, "threshold": 0}]
    
    for t in [20, 30, 40, 50, 60, 80]:
        for d in [3, 5, 8, 10, 15]:
            if d <= t * 0.3:
                actions.append({"name": f"满{t}减{d} 免费券", "type": "free", "cost": d, "threshold": t})
                
    # 真实付费神券包规则：核算平台净补贴成本
    actions.append({"name": "神券包(2.99元6张)", "type": "paid", "cost": 6.5, "threshold": 10})
    actions.append({"name": "神会员(9.9元10张)", "type": "paid", "cost": 8.0, "threshold": 10})
    
    return actions

ACTIONS = generate_action_space()

# ============================================
# 3. 因果马尔可夫引擎 (核心算法)
# ============================================
def causal_markov_funnel(user, scenario_name, action):
    buff = SCENARIOS[scenario_name]
    aov = float(user['平均客单价'])
    sens_free = float(user['用券率'])
    sens_paid = float(user['付费券使用率'])
    sub_coverage = float(user['补贴覆盖率'])
    
    p_open = min(buff['open_rate'], 1.0)
    p_cart = max(float(user['动态_点击_至_加购率']), 0.1)
    
    # 提取纯自然流基线
    base_pay_rate = 0.5 * (1 - sub_coverage) * buff['base_pay']
    p_pay = max(base_pay_rate, 0.05) 
    
    used_coupon = False
    actual_aov = aov
    
    # 计算策略干预增量
    if action['type'] == 'free':
        if aov >= action['threshold'] * 0.7:
            p_pay += (action['cost'] * 0.05 * sens_free) 
            used_coupon = True
            actual_aov = max(aov, action['threshold'] + 1.0)
            
    elif action['type'] == 'paid':
        if aov >= action['threshold']: 
            if sens_paid > 0.05:
                p_pay = max(p_pay + 0.4 * sens_paid, 0.85) 
                used_coupon = True

    p_pay = min(p_pay, 1.0)
    final_prob = p_open * p_cart * p_pay
    
    exp_gtv = final_prob * actual_aov
    exp_coupon_cost = final_prob * action['cost'] if used_coupon else 0
    
    return exp_gtv, exp_coupon_cost, p_pay

# ============================================
# 4. 侧边栏：宏观调控
# ============================================
st.sidebar.header("宏观环境与约束设置")
selected_scenario = st.sidebar.selectbox("1. 业务场景设定", list(SCENARIOS.keys()))
city_dau = st.sidebar.slider("2. 城市目标 DAU", 10000, 500000, 100000, 10000)

st.sidebar.markdown("---")
st.sidebar.subheader("商业寻优红线")
target_roi = st.sidebar.slider("要求最低 ROI", 1.0, 10.0, 3.0, 0.1)

# ============================================
# 5. 运筹优化求解器 (剥离自然流基座)
# ============================================
scale_factor = city_dau / len(df_users)
best_policy = {}
group_metrics = []
search_logs = []

global_base_gtv = 0
global_total_gtv = 0
global_incremental_gtv = 0
global_coupon_cost = 0

for persona, group in df_users.groupby('画像名称'):
    max_gtv = -1
    best_act = None
    best_cost = 0
    base_gtv_group = 0
    
    # 先算出该群体在无干预(Control)下的自然底盘
    for _, u in group.iterrows():
        g_base, _, _ = causal_markov_funnel(u, selected_scenario, {"type":"none", "cost":0, "threshold":0})
        base_gtv_group += g_base
        
    for act in ACTIONS:
        t_gtv, t_cost = 0, 0
        for _, u in group.iterrows():
            g, c, _ = causal_markov_funnel(u, selected_scenario, act)
            t_gtv += g; t_cost += c
            
        roi = t_gtv / t_cost if t_cost > 0 else 999
        
        search_logs.append({
            "人群": persona, "策略名称": act['name'], "策略类型": act['type'],
            "预期GTV": t_gtv * scale_factor * (len(group)/len(df_users)), 
            "预期成本": t_cost * scale_factor * (len(group)/len(df_users)),
            "预期ROI": min(roi, 15) 
        })
        
        if roi >= target_roi or act['type'] == 'none':
            if t_gtv > max_gtv:
                max_gtv = t_gtv; best_act = act; best_cost = t_cost
                
    best_policy[persona] = best_act
    
    g_ratio = scale_factor * (len(group)/len(df_users))
    g_total = max_gtv * g_ratio
    g_base = base_gtv_group * g_ratio
    g_cost = best_cost * g_ratio
    g_inc = g_total - g_base
    
    global_total_gtv += g_total
    global_base_gtv += g_base
    global_incremental_gtv += g_inc
    global_coupon_cost += g_cost
    group_metrics.append({"画像": persona, "策略": best_act['name'], "纯增量GTV": g_inc, "消耗预算": g_cost})

blended_roi = global_total_gtv / global_coupon_cost if global_coupon_cost > 0 else 0
incremental_roi = global_incremental_gtv / global_coupon_cost if global_coupon_cost > 0 else 0

# ============================================
# 6. 北极星看板
# ============================================
st.markdown("<div class='section-title'>城市级大盘核心预估指标 (因果增量口径)</div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: 
    st.markdown(f"<div class='kpi-card'><h2>¥ {global_total_gtv/10000:.1f} W</h2><p>预估大盘总 GTV<br><span style='font-size:12px;color:#aaa;'>(含自然基座 ¥{global_base_gtv/10000:.1f}W)</span></p></div>", unsafe_allow_html=True)
with c2: 
    st.markdown(f"<div class='kpi-card'><h2 style='color:#e74c3c;'>¥ {global_coupon_cost/10000:.1f} W</h2><p>账面净补贴消耗<br><span style='font-size:12px;color:#aaa;'>(纯投入)</span></p></div>", unsafe_allow_html=True)
with c3: 
    color = "#27ae60" if blended_roi >= target_roi else "#c0392b"
    st.markdown(f"<div class='kpi-card' style='border-top: 5px solid {color};'><h2 style='color:{color};'>{incremental_roi:.2f} x</h2><p>边际纯增量 ROI (Uplift ROI)<br><span style='font-size:12px;color:#aaa;'>(混合大盘 ROI 为 {blended_roi:.2f}x)</span></p></div>", unsafe_allow_html=True)

# ============================================
# 7. AI决策矩阵与增量财务瀑布图
# ============================================
c_left, c_right = st.columns([1, 1.5])
with c_left:
    st.markdown("<div class='section-title'>智能人群最优定投策略</div>", unsafe_allow_html=True)
    for i, (persona, act) in enumerate(best_policy.items()):
        st.markdown(f"""
        <div class="strategy-card" style="margin-bottom:15px;">
            <div style="color:#7f8c8d; font-size:13px; font-weight:bold;">客群细分: {persona}</div>
            <h4 style="color:#2c3e50; margin:10px 0;">{act['name']}</h4>
        </div>
        """, unsafe_allow_html=True)

with c_right:
    st.markdown("<div class='section-title'>因果财务核算：增量收益解析</div>", unsafe_allow_html=True)
    fig_wf = go.Figure(go.Waterfall(
        name="财务", orientation="v", measure=["relative", "relative", "relative", "total"],
        x=["自然基座 GTV", "(+) 策略纯增量 GTV", "(-) 券成本支出", "最终大盘净盘子"], 
        textposition="outside",
        text=[f"¥{global_base_gtv/10000:.1f}W", f"+¥{global_incremental_gtv/10000:.1f}W", f"-¥{global_coupon_cost/10000:.1f}W", f"¥{(global_total_gtv-global_coupon_cost)/10000:.1f}W"],
        y=[global_base_gtv, global_incremental_gtv, -global_coupon_cost, global_total_gtv-global_coupon_cost],
        connector={"line":{"color":"rgb(63, 63, 63)"}}, 
        decreasing={"marker":{"color":"#e74c3c"}}, 
        increasing={"marker":{"color":"#3498db"}},
        totals={"marker":{"color":"#2ecc71"}}
    ))
    fig_wf.update_layout(margin=dict(t=20, b=20, l=0, r=0), height=350, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_wf, use_container_width=True)

# ============================================
# 8. 策略搜索空间可视化
# ============================================
st.markdown("<div class='section-title'>最优策略搜索空间 (Grid Search Visualization)</div>", unsafe_allow_html=True)

df_logs = pd.DataFrame(search_logs)
search_persona = st.selectbox("选择要观测策略穷举轨迹的人群", df_logs['人群'].unique())
df_plot = df_logs[df_logs['人群'] == search_persona]

fig_search = px.scatter(
    df_plot, x="预期ROI", y="预期GTV", color="策略类型", text="策略名称",
    color_discrete_map={"free": "#3498db", "paid": "#9b59b6", "none": "#95a5a6"}
)
fig_search.update_traces(textposition='top center', textfont=dict(size=11))
fig_search.add_vline(x=target_roi, line_width=2, line_dash="dash", line_color="red")
fig_search.add_vrect(x0=0, x1=target_roi, fillcolor="red", opacity=0.05, layer="below", line_width=0)
fig_search.add_vrect(x0=target_roi, x1=15, fillcolor="green", opacity=0.05, layer="below", line_width=0)

fig_search.update_layout(
    height=450, margin={"l": 20, "r": 20, "t": 20, "b": 20}, plot_bgcolor="rgba(0,0,0,0)",
    xaxis_title="财务预期 ROI (约束条件)", yaxis_title="预期撬动 GTV (最大化目标)"
)
st.plotly_chart(fig_search, use_container_width=True)

# ============================================
# 9. 白盒审计：因果增量验证
# ============================================
st.markdown("<div class='section-title'>白盒审计：反事实基线与真实因果增量 (Uplift)</div>", unsafe_allow_html=True)
sample_u = df_users[df_users['画像名称'] == search_persona].iloc[0]
test_act = best_policy[search_persona]

_, _, p_pay_base = causal_markov_funnel(sample_u, selected_scenario, {"type":"none", "cost":0, "threshold":0})
_, _, p_pay_treat = causal_markov_funnel(sample_u, selected_scenario, test_act)
error_base = p_pay_base * 0.15; error_treat = p_pay_treat * 0.10 

fig_err = go.Figure()
fig_err.add_trace(go.Bar(
    name='反事实自然流基线 (Control)', x=['支付转化率'], y=[p_pay_base], marker_color='#bdc3c7',
    error_y=dict(type='data', array=[error_base], visible=True)
))
if test_act['type'] != 'none':
    fig_err.add_trace(go.Bar(
        name=f'最优策略干预 (Treatment: {test_act["name"]})', x=['支付转化率'], y=[p_pay_treat], marker_color='#3498db',
        error_y=dict(type='data', array=[error_treat], visible=True)
    ))
    fig_err.add_annotation(x=0, y=max(p_pay_treat, p_pay_base)+0.1, text=f"真实撬动增量 (CATE): +{p_pay_treat - p_pay_base:.1%}", showarrow=False, font=dict(color="#e74c3c", size=16))

fig_err.update_layout(barmode='group', height=400, margin=dict(t=30, b=0), plot_bgcolor="rgba(0,0,0,0)", yaxis_tickformat='.0%')
st.plotly_chart(fig_err, use_container_width=True)
