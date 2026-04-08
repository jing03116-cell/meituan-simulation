import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="大盘全量补贴策略寻优引擎", layout="wide")
st.markdown("""
<style>
    .big-metric { background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; border-bottom: 4px solid #1f77b4; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    .big-metric h2 { color: #1f77b4; margin: 0; font-size: 36px; }
    .big-metric p { color: #666; margin: 5px 0 0 0; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

st.title("🌍 城市级大盘仿真与【全域人群归因】沙盘")
st.markdown("不仅能搜出全局最优解，更能一键拆解**不同业务场景下的人群贡献度（GTV 归因分析）**。")

# ==========================================
# 1. 数据基建加载
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("大宽表.csv")
    except:
        data = []
        personas = ["高客单品质/家庭党", "极致神券外卖羊毛党", "线下到店体验/钝感党", "早午餐刚需/白嫖党"]
        for i in range(200):
            idx = i % 4
            data.append({
                "user_id": f"U{i}", "画像名称": personas[idx], 
                "平均客单价": [68.0, 35.0, 55.0, 28.0][idx] + np.random.normal(0, 5),
                "用券率": [0.4, 0.9, 0.2, 0.8][idx], 
                "付费券使用率": [0.3, 0.6, 0.1, 0.1][idx],
                "动态_下午茶活跃度": [0.8, 0.1, 0.3, 0.4][idx],
                "动态_点击_至_加购率": [0.6, 0.8, 0.3, 0.7][idx],
                "补贴覆盖率": [0.3, 0.95, 0.1, 0.8][idx]
            })
        df = pd.DataFrame(data)
    for col in ['平均客单价', '用券率', '付费券使用率', '动态_点击_至_加购率', '补贴覆盖率']:
        if col in df.columns: df[col] = df[col].fillna(df[col].mean())
    return df

df_users = load_data()

SCENARIOS = {
    "☕ 周末下午茶": {"open_app": 1.2, "cart_to_pay": 1.1},
    "🌧️ 暴雨晚高峰": {"open_app": 1.8, "cart_to_pay": 0.5},
    "🚗 节假日出行": {"open_app": 1.4, "cart_to_pay": 1.2}
}

# ==========================================
# 2. 动态生成动作空间
# ==========================================
@st.cache_data
def generate_action_space():
    actions = [{"name": "🚫 不发券", "type": "none", "cost": 0, "threshold": 0}]
    thresholds = [20, 25, 30, 35, 40, 50, 60]
    discounts = [2, 3, 4, 5, 6, 8, 10, 12]
    for t in thresholds:
        for d in discounts:
            if d <= t * 0.3:
                actions.append({"name": f"🎫 满{t}减{d}", "type": "free", "cost": d, "threshold": t})
    actions.append({"name": "💎 1.9元神券包(低膨胀)", "type": "paid", "cost": 4, "threshold": 0})
    actions.append({"name": "💎 5.9元神券包(高膨胀)", "type": "paid", "cost": 10, "threshold": 0})
    return actions

ACTIONS = generate_action_space()

# ==========================================
# 3. 核心马尔可夫引擎
# ==========================================
def markov_funnel(user, scenario_name, action):
    buff = SCENARIOS[scenario_name]
    aov = float(user['平均客单价'])
    sens_free = float(user['用券率'])
    sens_paid = float(user['付费券使用率'])
    sub_coverage = float(user.get('补贴覆盖率', 0.5))
    
    p_open = min(0.4 * buff['open_app'], 1.0)
    p_cart = max(float(user['动态_点击_至_加购率']), 0.1)
    base_pay_rate = 0.5 * (1 - sub_coverage) * buff['cart_to_pay']
    p_pay = max(base_pay_rate, 0.05) 
    
    used_coupon = False
    actual_aov = aov
    
    if action['type'] == 'free':
        if aov >= action['threshold'] * 0.7:
            p_pay += (action['cost'] * 0.06 * sens_free)
            used_coupon = True
            actual_aov = max(aov, action['threshold'] + 1.0)
    elif action['type'] == 'paid':
        if sens_paid > 0.05:
            p_pay = max(p_pay + 0.4 * sens_paid, 0.8)
            used_coupon = True

    p_pay = min(p_pay, 1.0)
    final_prob = p_open * p_cart * p_pay
    return final_prob * actual_aov, final_prob * action['cost'] if used_coupon else 0

# ==========================================
# 4. 控制台
# ==========================================
st.sidebar.header("🛠️ 大盘调控与场景切换")
selected_scenario = st.sidebar.selectbox("1. 业务场景设定", list(SCENARIOS.keys()))
city_dau = st.sidebar.slider("2. 目标城市日活 (DAU)", 10000, 500000, 100000, 10000)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 AI 寻优边界条件")
min_roi_constraint = st.sidebar.slider("3. 财务底线 ROI 约束", 1.0, 8.0, 2.5, 0.1)

# ==========================================
# 5. 全维网格寻优与【归因数据收集】
# ==========================================
scale_factor = city_dau / len(df_users)
best_policy = {}
attribution_logs = [] # 核心新增：用于记录归因分析的数据

global_total_gtv = 0
global_total_cost = 0

for persona, group in df_users.groupby('画像名称'):
    max_gtv = -1
    best_act = None
    best_act_cost = 0
    
    for act in ACTIONS:
        tot_gtv, tot_cost = 0, 0
        for _, u in group.iterrows():
            gtv, cost = markov_funnel(u, selected_scenario, act)
            tot_gtv += gtv; tot_cost += cost
            
        roi = tot_gtv / tot_cost if tot_cost > 0 else 999
        
        if roi >= min_roi_constraint or act['type'] == 'none':
            if tot_gtv > max_gtv:
                max_gtv = tot_gtv
                best_act = act
                best_act_cost = tot_cost
                
    best_policy[persona] = best_act
    
    # 记录该群体在当前场景、最优策略下的大盘绝对值贡献
    group_scaled_gtv = max_gtv * scale_factor * (len(group)/len(df_users))
    group_scaled_cost = best_act_cost * scale_factor * (len(group)/len(df_users))
    
    attribution_logs.append({
        "群体画像": persona,
        "分配策略": best_act['name'],
        "贡献GTV": group_scaled_gtv,
        "消耗补贴成本": group_scaled_cost,
        "群体ROI": group_scaled_gtv / group_scaled_cost if group_scaled_cost > 0 else 0
    })
    
    global_total_gtv += group_scaled_gtv
    global_total_cost += group_scaled_cost

global_roi = global_total_gtv / global_total_cost if global_total_cost > 0 else 0

# ==========================================
# 6. 大盘结果展板
# ==========================================
st.subheader(f"📊 【{selected_scenario}】城市级全局大盘预期")
col1, col2, col3 = st.columns(3)
with col1: st.markdown(f"<div class='big-metric'><h2>¥ {global_total_gtv:,.0f}</h2><p>🚀 全局加总预期 GTV</p></div>", unsafe_allow_html=True)
with col2: st.markdown(f"<div class='big-metric'><h2 style='color:#d32f2f;'>¥ {global_total_cost:,.0f}</h2><p>📉 全局大盘补贴总消耗</p></div>", unsafe_allow_html=True)
with col3: 
    color = "green" if global_roi >= min_roi_constraint else "red"
    st.markdown(f"<div class='big-metric'><h2 style='color:{color};'>{global_roi:.2f} x</h2><p>⚖️ 最终全局核算 ROI</p></div>", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 7. 核心新增：场景归因与群体贡献拆解
# ==========================================
st.subheader("🥧 场景 GTV 归因分析 (是哪群人撑起了这个场景？)")
st.markdown("通过下方图表，可以清晰洞察在当前场景下，**平台大盘的 GTV 结构分布**以及**补贴资金的流向效率**。")

df_attr = pd.DataFrame(attribution_logs)

col_pie, col_bar = st.columns([1, 1.5])

with col_pie:
    # 甜甜圈图：结构占比归因
    fig_pie = px.pie(
        df_attr, 
        values='贡献GTV', 
        names='群体画像', 
        hole=0.4,
        title=f"【{selected_scenario}】GTV 贡献来源占比",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bar:
    # 双轴对比图：看哪个人群“花钱少办事多”
    fig_bar = go.Figure()
    # 添加 GTV 柱子
    fig_bar.add_trace(go.Bar(
        x=df_attr['群体画像'], y=df_attr['贡献GTV'], name='带来 GTV (收益)', marker_color='#64b5f6', text=df_attr['分配策略']
    ))
    # 添加 成本 柱子
    fig_bar.add_trace(go.Bar(
        x=df_attr['群体画像'], y=df_attr['消耗补贴成本'], name='消耗成本 (投入)', marker_color='#e57373'
    ))
    fig_bar.update_layout(
        title="各群体投入产出比 (Hover 查看分配的具体策略)",
        barmode='group',
        margin=dict(t=40, b=0, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# 动态业务洞察点评 (AI 自动生成结论)
top_gtv_group = df_attr.loc[df_attr['贡献GTV'].idxmax()]['群体画像']
top_cost_group = df_attr.loc[df_attr['消耗补贴成本'].idxmax()]['群体画像']

st.info(f"💡 **AI 场景归因洞察：** 在【{selected_scenario}】场景下，大盘核心驱动力来自 **{top_gtv_group}**；同时需注意，资金消耗大头集中在 **{top_cost_group}** 群体。")
