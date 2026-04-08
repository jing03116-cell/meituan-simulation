"""
AI User Simulation & Smart Subsidy Oracle
Enterprise Production Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 0. UI Configuration
# ============================================
st.set_page_config(
    page_title="Smart Subsidy Oracle",
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

st.title("Smart Subsidy Oracle (Causal-OR Engine)")
st.markdown("Based on Causal Inference and MCKP (Multiple Choice Knapsack Problem) Optimization. Simultaneously evaluates Free and Paid coupons under strict Global Budget and ROI constraints.")

# ============================================
# 1. Data Contract
# ============================================
@st.cache_data
def load_real_user_data():
    try:
        df = pd.read_csv("大宽表.csv")
    except FileNotFoundError:
        st.error("Error: [大宽表.csv] not found in the current directory.")
        st.stop()

    mapping = {
        "画像名称": "persona",
        "平均客单价": "avg_order_value",
        "补贴覆盖率": "sub_coverage",      
        "用券率": "coupon_sensitivity",         
        "付费券使用率": "paid_sensitivity",
        "动态_点击_至_加购率": "base_conversion_rate" 
    }
    
    if "user_id" not in df.columns:
        df["user_id"] = ["U" + str(i).zfill(6) for i in range(len(df))]
        
    df = df.rename(columns=mapping)
    
    for col in ['avg_order_value', 'sub_coverage', 'coupon_sensitivity', 'paid_sensitivity', 'base_conversion_rate']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
            
    if 'avg_order_value' in df.columns:
        df['avg_order_value'] = df['avg_order_value'].clip(lower=15)
        
    return df

df_users = load_real_user_data()

SCENARIOS = {
    "Weekend_Afternoon_Tea": {"open_rate": 0.45, "base_pay": 0.35},
    "Rainy_Night": {"open_rate": 0.65, "base_pay": 0.20}, 
    "Holiday_Travel": {"open_rate": 0.55, "base_pay": 0.45}
}

# ============================================
# 2. Mixed Action Space (Free & Paid)
# ============================================
@st.cache_data
def generate_action_space():
    actions = [{"name": "Control (No Coupon)", "type": "none", "cost": 0, "threshold": 0, "upfront": 0}]
    
    # Free Coupons
    for t in [20, 30, 40, 50, 60, 80]:
        for d in [3, 5, 8, 10, 15]:
            if d <= t * 0.3:
                actions.append({"name": f"Free: {t}-{d}", "type": "free", "cost": d, "threshold": t, "upfront": 0})
                
    # Paid Coupons (e.g. 2.99 for 6 coupons -> 0.5 upfront per use. Platform net cost = expansion - upfront)
    actions.append({"name": "Paid: 6-Pack (Net 6.5)", "type": "paid", "cost": 6.5, "threshold": 10, "upfront": 0.5})
    actions.append({"name": "Paid: VIP (Net 8.0)", "type": "paid", "cost": 8.0, "threshold": 10, "upfront": 1.0})
    
    return actions

ACTIONS = generate_action_space()

# ============================================
# 3. Vectorized Causal Engine (High Performance)
# ============================================
def eval_action_vectorized(df_group, scenario_name, action):
    buff = SCENARIOS[scenario_name]
    
    aov = df_group['avg_order_value'].values
    sens_free = df_group['coupon_sensitivity'].values
    sens_paid = df_group['paid_sensitivity'].values
    sub_cov = df_group['sub_coverage'].values
    p_cart = df_group['base_conversion_rate'].values
    
    p_open = np.minimum(buff['open_rate'], 1.0)
    p_cart = np.maximum(p_cart, 0.1)
    
    # Counterfactual Baseline
    base_pay_rate = 0.5 * (1 - sub_cov) * buff['base_pay']
    p_pay_base = np.maximum(base_pay_rate, 0.05)
    
    p_pay_treat = p_pay_base.copy()
    used_coupon = np.zeros(len(df_group), dtype=bool)
    actual_aov = aov.copy()
    upfront_rev = np.zeros(len(df_group))
    
    # Treatment Uplift
    if action['type'] == 'free':
        mask = aov >= action['threshold'] * 0.7
        p_pay_treat[mask] += (action['cost'] * 0.05 * sens_free[mask])
        used_coupon = mask
        actual_aov[mask] = np.maximum(aov[mask], action['threshold'] + 1.0)
        
    elif action['type'] == 'paid':
        mask = (aov >= action['threshold']) & (sens_paid > 0.05)
        p_pay_treat[mask] = np.maximum(p_pay_treat[mask] + 0.4 * sens_paid[mask], 0.85)
        used_coupon = mask
        upfront_rev[mask] = action['upfront']
        
    p_pay_treat = np.clip(p_pay_treat, 0.0, 1.0)
    
    final_prob_base = p_open * p_cart * p_pay_base
    final_prob_treat = p_open * p_cart * p_pay_treat
    
    exp_gtv_base = final_prob_base * aov
    # GTV includes actual order value + upfront package purchase price
    exp_gtv_treat = final_prob_treat * actual_aov + final_prob_treat * upfront_rev
    exp_cost_treat = np.where(used_coupon, final_prob_treat * action['cost'], 0)
    
    # Deadweight Loss (Cost spent on users who would have converted anyway)
    safe_treat = np.where(p_pay_treat > 0, p_pay_treat, 1)
    dwl_ratio = np.where(p_pay_treat > 0, p_pay_base / safe_treat, 0)
    exp_dwl = np.where(used_coupon, final_prob_treat * dwl_ratio * action['cost'], 0)
    
    return exp_gtv_base.sum(), exp_gtv_treat.sum(), exp_cost_treat.sum(), exp_dwl.sum()

# ============================================
# 4. Sidebar: Global Constraints
# ============================================
st.sidebar.header("Global Constraints")
selected_scenario = st.sidebar.selectbox("Scenario", list(SCENARIOS.keys()))
city_dau = st.sidebar.slider("City DAU", 10000, 500000, 100000, 10000)

st.sidebar.markdown("---")
st.sidebar.subheader("Optimization Constraints")
global_budget = st.sidebar.slider("Global Budget Cap (RMB)", 10000, 500000, 150000, 10000)
target_roi = st.sidebar.slider("Target Uplift ROI", 1.0, 10.0, 3.0, 0.1)

# ============================================
# 5. AI MCKP Knapsack Solver
# ============================================
scale_factor = city_dau / len(df_users)
segments_data = list(df_users.groupby('persona'))

candidates_matrix = []
base_gtv_global = 0

# Phase 1: Evaluate all actions for all segments
for persona, group in segments_data:
    g_base_raw, _, _, _ = eval_action_vectorized(group, selected_scenario, ACTIONS[0])
    g_base_scaled = g_base_raw * scale_factor
    base_gtv_global += g_base_scaled
    
    seg_candidates = []
    for act in ACTIONS:
        _, g_treat, c_treat, dwl = eval_action_vectorized(group, selected_scenario, act)
        g_treat_scaled = g_treat * scale_factor
        c_treat_scaled = c_treat * scale_factor
        dwl_scaled = dwl * scale_factor
        
        inc_gtv = g_treat_scaled - g_base_scaled
        inc_cost = c_treat_scaled
        
        roi = inc_gtv / inc_cost if inc_cost > 0 else (999.0 if inc_gtv >= 0 else -999.0)
        
        if roi >= target_roi or act['type'] == 'none':
            seg_candidates.append({
                "persona": persona, "act": act, "inc_gtv": inc_gtv, 
                "inc_cost": inc_cost, "roi": roi, "dwl": dwl_scaled, "total_gtv": g_treat_scaled
            })
            
    # Sort descending by GTV to establish local optimum
    seg_candidates.sort(key=lambda x: x['inc_gtv'], reverse=True)
    candidates_matrix.append(seg_candidates)

# Phase 2: AI Downgrade Algorithm (Resolve Global Budget Cap)
current_picks = {i: 0 for i in range(len(candidates_matrix))}

while True:
    current_total_cost = sum(candidates_matrix[i][current_picks[i]]['inc_cost'] for i in range(len(candidates_matrix)))
    if current_total_cost <= global_budget:
        break 
        
    worst_i = -1
    worst_roi = float('inf')
    
    # Find the segment with the lowest ROI to downgrade
    for i in range(len(candidates_matrix)):
        pick_idx = current_picks[i]
        cand = candidates_matrix[i][pick_idx]
        if cand['inc_cost'] > 0 and cand['roi'] < worst_roi:
            if pick_idx + 1 < len(candidates_matrix[i]): 
                worst_roi = cand['roi']
                worst_i = i
                
    if worst_i == -1:
        break 
        
    current_picks[worst_i] += 1

# Phase 3: Compile Results
best_policy = {}
global_total_gtv = 0
global_coupon_cost = 0
global_dwl = 0

for i in range(len(candidates_matrix)):
    final_cand = candidates_matrix[i][current_picks[i]]
    best_policy[final_cand['persona']] = final_cand['act']
    global_total_gtv += final_cand['total_gtv']
    global_coupon_cost += final_cand['inc_cost']
    global_dwl += final_cand['dwl']

global_incremental_gtv = global_total_gtv - base_gtv_global
incremental_roi = global_incremental_gtv / global_coupon_cost if global_coupon_cost > 0 else 0
upfront_revenue = sum(candidates_matrix[i][current_picks[i]]['act']['upfront'] * (candidates_matrix[i][current_picks[i]]['total_gtv'] / (df_users['avg_order_value'].mean() + candidates_matrix[i][current_picks[i]]['act']['upfront'])) for i in range(len(candidates_matrix)) if candidates_matrix[i][current_picks[i]]['act']['type'] == 'paid') # Approx estimation for visualization

# ============================================
# 6. Executive Dashboard
# ============================================
st.markdown("<div class='section-title'>Global Metrics (Causal Uplift & Budget Constrained)</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1: 
    st.markdown(f"<div class='kpi-card'><h2>{global_total_gtv/10000:.1f} W</h2><p>Total GTV<br><span style='font-size:12px;color:#aaa;'>(Base: {base_gtv_global/10000:.1f}W)</span></p></div>", unsafe_allow_html=True)
with c2: 
    color = "#e74c3c" if global_coupon_cost >= global_budget * 0.95 else "#3498db"
    st.markdown(f"<div class='kpi-card'><h2 style='color:{color};'>{global_coupon_cost/10000:.1f} W</h2><p>Total Cost<br><span style='font-size:12px;color:#aaa;'>(Cap: {global_budget/10000:.1f}W)</span></p></div>", unsafe_allow_html=True)
with c3: 
    st.markdown(f"<div class='kpi-card'><h2 style='color:#e67e22;'>{global_dwl/10000:.1f} W</h2><p>Deadweight Loss<br><span style='font-size:12px;color:#aaa;'>(Free-rider Cost)</span></p></div>", unsafe_allow_html=True)
with c4: 
    color = "#27ae60" if incremental_roi >= target_roi else "#c0392b"
    st.markdown(f"<div class='kpi-card' style='border-top: 5px solid {color};'><h2 style='color:{color};'>{incremental_roi:.2f} x</h2><p>Uplift ROI</p></div>", unsafe_allow_html=True)

# ============================================
# 7. AI Mixed-Strategy Matrix & Waterfall
# ============================================
c_left, c_right = st.columns([1, 1.8])
with c_left:
    st.markdown("<div class='section-title'>AI Optimal Portfolio</div>", unsafe_allow_html=True)
    for persona, act in best_policy.items():
        st.markdown(f"""
        <div class="strategy-card" style="margin-bottom:15px;">
            <div style="color:#7f8c8d; font-size:13px; font-weight:bold;">Segment: {persona}</div>
            <h4 style="color:#2c3e50; margin:10px 0;">{act['name']}</h4>
        </div>
        """, unsafe_allow_html=True)

with c_right:
    st.markdown("<div class='section-title'>Causal Financial Waterfall</div>", unsafe_allow_html=True)
    pure_inc_gtv = global_incremental_gtv - upfront_revenue
    
    fig_wf = go.Figure(go.Waterfall(
        name="Financials", orientation="v", measure=["relative", "relative", "relative", "relative", "total"],
        x=["Base GTV", "(+) Upfront Revenue", "(+) Pure Uplift GTV", "(-) Sub Cost", "Net Output"], 
        textposition="outside",
        text=[f"{base_gtv_global/10000:.1f}W", f"+{upfront_revenue/10000:.1f}W", f"+{pure_inc_gtv/10000:.1f}W", f"-{global_coupon_cost/10000:.1f}W", f"{(global_total_gtv-global_coupon_cost)/10000:.1f}W"],
        y=[base_gtv_global, upfront_revenue, pure_inc_gtv, -global_coupon_cost, global_total_gtv-global_coupon_cost],
        connector={"line":{"color":"rgb(63, 63, 63)"}}, 
        decreasing={"marker":{"color":"#e74c3c"}}, 
        increasing={"marker":{"color":"#3498db"}},
        totals={"marker":{"color":"#2ecc71"}}
    ))
    fig_wf.update_layout(margin=dict(t=20, b=20, l=0, r=0), height=400, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_wf, use_container_width=True)
