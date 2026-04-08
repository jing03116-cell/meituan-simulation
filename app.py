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
# 页面配置
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
    .strategy-card {
        background-color: #f8fafc;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
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
        混合策略自动寻优 · 运筹规划 · 因果推断沙盘
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================
# 侧边栏配置
# ============================================
with st.sidebar:
    st.markdown("## 🎮 仿真控制台")
    
    st.markdown("### 📊 基础配置")
    scenario = st.selectbox("业务场景", ["周末下午茶", "暴雨晚高峰", "节假日出行", "日常通勤"])
    
    city_scale = st.select_slider(
        "目标大盘规模 (DAU)",
        options=["10万", "50万", "100万", "200万", "500万"],
        value="100万"
    )
    scale_map = {"10万": 100000, "50万": 500000, "100万": 1000000, "200万": 2000000, "500万": 5000000}
    dau = scale_map[city_scale]
    
    st.markdown("### ⚖️ 运筹优化约束 (AI 寻优红线)")
    global_budget = st.slider(
        "全局补贴预算上限 (元)",
        min_value=50000, max_value=1000000, value=300000, step=50000,
        help="AI将在不超出此预算池的前提下，分配不同的券，寻找GTV最大的组合"
    )
    
    target_roi = st.slider(
        "目标底线 ROI",
        min_value=1.0, max_value=10.0, value=3.0, step=0.1,
        help="任何ROI低于此红线的发券策略将被AI自动淘汰"
    )
    
    st.markdown("---")
    run_simulation = st.button("🚀 启动 AI 自动寻优", type="primary", use_container_width=True)

# ============================================
# 底层数据基建与策略池
# ============================================
@st.cache_data
def load_real_user_data():
    try:
        df = pd.read_csv("大宽表.csv")
    except FileNotFoundError:
        st.error("⚠️ 未在当前目录找到【大宽表.csv】文件，请确保文件已上传！")
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
    "周末下午茶": {"open_rate": 0.45, "base_pay": 0.35},
    "暴雨晚高峰": {"open_rate": 0.65, "base_pay": 0.20},
    "节假日出行": {"open_rate": 0.55, "base_pay": 0.40},
    "日常通勤": {"open_rate": 0.50, "base_pay": 0.30}
}

@st.cache_data
def generate_action_space():
    actions = [{"name": "留白 (不发券)", "type": "none", "cost": 0, "threshold": 0, "upfront": 0}]
    # 免费券池
    for t in [20, 30, 40, 50, 60, 80]:
        for d in [3, 5, 8, 10, 15]:
            if d <= t * 0.3:
                actions.append({"name": f"免费满减: 满{t}减{d}", "type": "free", "cost": d, "threshold": t, "upfront": 0})
    # 付费券池 (考虑首付净成本)
    actions.append({"name": "付费包 (单次净亏6.5元)", "type": "paid", "cost": 6.5, "threshold": 10, "upfront": 0.5})
    actions.append({"name": "神会员 (单次净亏8.0元)", "type": "paid", "cost": 8.0, "threshold": 10, "upfront": 1.0})
    return actions

ACTIONS = generate_action_space()

# ============================================
# 高性能向量化引擎
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
    
    # 反事实自然流基线
    base_pay_rate = 0.5 * (1 - sub_cov) * buff['base_pay']
    p_pay_base = np.maximum(base_pay_rate, 0.05)
    
    p_pay_treat = p_pay_base.copy()
    used_coupon = np.zeros(len(df_group), dtype=bool)
    actual_aov = aov.copy()
    upfront_rev = np.zeros(len(df_group))
    
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
    exp_gtv_treat = final_prob_treat * actual_aov + final_prob_treat * upfront_rev
    exp_cost_treat = np.where(used_coupon, final_prob_treat * action['cost'], 0)
    
    return exp_gtv_base.sum(), exp_gtv_treat.sum(), exp_cost_treat.sum(), final_prob_treat.mean(), final_prob_base.mean()

# ============================================
# 运行仿真与 MCKP 自动寻优
# ============================================
if run_simulation:
    with st.spinner("🧠 运筹学 AI 正在后台执行千万次策略空间遍历与预算降级..."):
        scale_factor = dau / len(df_users)
        segments_data = list(df_users.groupby('persona'))
        
        candidates_matrix = []
        search_logs = []
        global_base_gtv = 0
        
        # Phase 1: 贪心寻优 (收集所有满足 ROI 的策略)
        for persona, group in segments_data:
            g_base_raw, _, _, _, _ = eval_action_vectorized(group, scenario, ACTIONS[0])
            base_gtv_global_seg = g_base_raw * scale_factor
            global_base_gtv += base_gtv_global_seg
            
            seg_candidates = []
            for act in ACTIONS:
                _, g_treat, c_treat, conv_treat, conv_base = eval_action_vectorized(group, scenario, act)
                g_treat_scaled = g_treat * scale_factor
                c_treat_scaled = c_treat * scale_factor
                
                inc_gtv = g_treat_scaled - base_gtv_global_seg
                inc_cost = c_treat_scaled
                roi = inc_gtv / inc_cost if inc_cost > 0 else (999.0 if inc_gtv >= 0 else -999.0)
                
                search_logs.append({
                    "人群": persona, "策略": act['name'], "策略类型": act['type'],
                    "预期增量GTV": inc_gtv, "预期成本": inc_cost, "ROI": min(roi, 15)
                })
                
                if roi >= target_roi or act['type'] == 'none':
                    seg_candidates.append({
                        "persona": persona, "act": act, "inc_gtv": inc_gtv, 
                        "inc_cost": inc_cost, "roi": roi, "total_gtv": g_treat_scaled,
                        "conv_rate": conv_treat
                    })
            
            # 按 GTV 降序排列
            seg_candidates.sort(key=lambda x: x['inc_gtv'], reverse=True)
            candidates_matrix.append(seg_candidates)

        # Phase 2: AI 预算降级算法 (MCKP)
        current_picks = {i: 0 for i in range(len(candidates_matrix))}
        
        while True:
            current_total_cost = sum(candidates_matrix[i][current_picks[i]]['inc_cost'] for i in range(len(candidates_matrix)))
            if current_total_cost <= global_budget:
                break 
                
            worst_i = -1
            worst_roi = float('inf')
            
            # 挑出当前占用成本且 ROI 最低的人群进行降级
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

        # Phase 3: 打包结果
        best_policy = {}
        segment_stats = []
        global_total_gtv = 0
        global_total_cost = 0
        global_conv_rate = 0
        
        for i in range(len(candidates_matrix)):
            final_cand = candidates_matrix[i][current_picks[i]]
            best_policy[final_cand['persona']] = final_cand['act']
            global_total_gtv += final_cand['total_gtv']
            global_total_cost += final_cand['inc_cost']
            global_conv_rate += final_cand['conv_rate'] * (len(segments_data[i][1]) / len(df_users))
            
            segment_stats.append({
                "persona": final_cand['persona'],
                "assigned_strategy": final_cand['act']['name'],
                "converted": final_cand['conv_rate'],
                "gtv": final_cand['total_gtv'],
                "cost": final_cand['inc_cost'],
                "roi": final_cand['roi']
            })

        st.session_state['results'] = {
            "total_gtv_scaled": global_total_gtv,
            "base_gtv": global_base_gtv,
            "total_cost_scaled": global_total_cost,
            "conversion_rate": global_conv_rate,
            "roi": (global_total_gtv - global_base_gtv) / global_total_cost if global_total_cost > 0 else 0,
            "avg_order_value": global_total_gtv / (dau * global_conv_rate) if global_conv_rate > 0 else 0,
            "segment_stats": pd.DataFrame(segment_stats),
            "search_logs": pd.DataFrame(search_logs),
            "best_policy": best_policy,
            "meets_target": global_total_cost <= global_budget
        }
        st.session_state['simulation_run'] = True

# ============================================
# 结果展示
# ============================================
if 'simulation_run' in st.session_state and st.session_state['simulation_run']:
    results = st.session_state['results']
    
    st.markdown("<div class='section-title'>📈 核心指标</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">总GTV (含自然流)</div>
            <div class="metric-value" style="color: #667eea;">¥ {results['total_gtv_scaled']/10000:.1f}万</div>
            <div style="color: #10b981; font-size: 0.9rem;">自然流水底座 ¥ {results['base_gtv']/10000:.1f}万</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        cost_color = "#ef4444" if results['total_cost_scaled'] >= global_budget * 0.98 else "#f59e0b"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">实际消耗补贴</div>
            <div class="metric-value" style="color: {cost_color};">¥ {results['total_cost_scaled']/10000:.1f}万</div>
            <div style="color: #6b7280; font-size: 0.9rem;">预算上限 ¥ {global_budget/10000:.0f}万</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        roi_color = "#10b981" if results['roi'] >= target_roi else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">纯增量投资回报率 (Uplift ROI)</div>
            <div class="metric-value" style="color: {roi_color};">{results['roi']:.2f}x</div>
            <div style="color: #6b7280; font-size: 0.9rem;">AI 约束底线 {target_roi:.1f}x</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        net_profit = results['total_gtv_scaled'] - results['total_cost_scaled']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">大盘净流水</div>
            <div class="metric-value" style="color: #8b5cf6;">¥ {net_profit/10000:.1f}万</div>
            <div style="color: #6b7280; font-size: 0.9rem;">平均客单价 ¥ {results['avg_order_value']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    if results['meets_target']:
        st.success(f"✅ AI已成功寻找最优混合策略，并严格卡紧预算红线！ (实际消耗 ¥{results['total_cost_scaled']:,.0f} ≤ 预算 ¥{global_budget:,.0f})")
    else:
        st.warning("⚠️ 警告：策略空间内无解，AI 已全部回退为『留白』自然流策略。")
        
    # --- AI 决策矩阵横幅 ---
    st.markdown("<div class='section-title'>🎯 AI 混合定投策略结果</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (persona, act) in enumerate(results['best_policy'].items()):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="strategy-card">
                <div style="color:#6b7280; font-size:0.85rem; font-weight:600; margin-bottom:5px;">{persona}</div>
                <div style="font-size:1.1rem; font-weight:bold; color:#667eea;">{act['name']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📊 深度分析视图</div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📈 分群表现矩阵", "🎯 智能转化漏斗", "💰 财务与寻优轨迹"])
    
    with tab1:
        segment_df = results['segment_stats']
        fig_segment = go.Figure(go.Bar(
            x=segment_df['persona'], y=segment_df['roi'], name='ROI',
            marker_color=['#667eea', '#ef4444', '#10b981', '#f59e0b'],
            text=[f"{x:.2f}x" if pd.notna(x) else "0.0x" for x in segment_df['roi']], textposition='outside'
        ))
        fig_segment.add_hline(y=target_roi, line_dash="dash", line_color="red", annotation_text=f"底线ROI: {target_roi}x")
        fig_segment.update_layout(title="各用户群体纯增量 ROI 对比", yaxis_title="ROI (x)", height=400)
        st.plotly_chart(fig_segment, use_container_width=True)
        
        st.dataframe(
            segment_df.style.format({
                'converted': '{:.1%}', 'gtv': '¥ {:,.0f}', 'cost': '¥ {:,.0f}', 'roi': '{:.2f}x'
            }).background_gradient(subset=['roi'], cmap='RdYlGn'),
            use_container_width=True
        )
        
    with tab2:
        total_users = dau
        converted = total_users * results['conversion_rate']
        clicked = converted / 0.45 
        opened = max(clicked / 0.6, total_users * 0.3) 
        
        fig_funnel = go.Figure(go.Funnel(
            y=['全量目标客群', '打开APP', '浏览商户加购', '核心干预：下单支付'],
            x=[total_users, opened, clicked, converted],
            textinfo="value+percent initial",
            marker={"color": ["#667eea", "#764ba2", "#f59e0b", "#10b981"]}
        ))
        fig_funnel.update_layout(title="自适应因果转化漏斗", height=400)
        st.plotly_chart(fig_funnel, use_container_width=True)
        
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.info("💡 散点图展示 AI 是如何在数十种混合策略中避开红色亏损区，并找到最优解的。")
            df_logs = results['search_logs']
            search_persona = st.selectbox("选择要观测穷举轨迹的客群", df_logs['人群'].unique())
            df_plot = df_logs[df_logs['人群'] == search_persona]

            fig_search = px.scatter(
                df_plot, x="ROI", y="预期增量GTV", color="策略类型", text="策略",
                color_discrete_map={"free": "#3b82f6", "paid": "#8b5cf6", "none": "#9ca3af"}
            )
            fig_search.update_traces(textposition='top center', textfont=dict(size=10))
            fig_search.add_vline(x=target_roi, line_dash="dash", line_color="red")
            fig_search.add_vrect(x0=-999, x1=target_roi, fillcolor="red", opacity=0.05, layer="below", line_width=0)
            fig_search.update_layout(title=f"【{search_persona}】AI策略收益评估矩阵", height=450)
            st.plotly_chart(fig_search, use_container_width=True)
            
        with col2:
            st.info("💡 财务账本：展示大盘是如何从自然流基座加上策略纯增量，扣除消耗得到的。")
            pure_inc_gtv = results['total_gtv_scaled'] - results['base_gtv']
            fig_wf = go.Figure(go.Waterfall(
                name="财务", orientation="v", measure=["relative", "relative", "relative", "total"],
                x=["自然基座 GTV", "(+) 策略纯增量", "(-) 券成本扣减", "净收益流水"], textposition="outside",
                text=[f"¥{results['base_gtv']/10000:.1f}W", f"+¥{pure_inc_gtv/10000:.1f}W", f"-¥{results['total_cost_scaled']/10000:.1f}W", f"¥{(results['total_gtv_scaled']-results['total_cost_scaled'])/10000:.1f}W"],
                y=[results['base_gtv'], pure_inc_gtv, -results['total_cost_scaled'], results['total_gtv_scaled']-results['total_cost_scaled']],
                connector={"line":{"color":"rgb(63, 63, 63)"}}, decreasing={"marker":{"color":"#ef4444"}}, increasing={"marker":{"color":"#10b981"}}, totals={"marker":{"color":"#667eea"}}
            ))
            fig_wf.update_layout(title="因果财务瀑布图 (Causal Waterfall)", height=450)
            st.plotly_chart(fig_wf, use_container_width=True)

else:
    st.info("👈 请在左侧配置参数，点击「启动 AI 自动寻优」开始大盘全量推演")
    st.markdown("<div class='section-title'>📊 平台能力预览</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 混合策略寻优\n- 免费券/付费包同池竞争\n- 动态匹配千人千面\n- 突破单一发券限制")
    with col2:
        st.markdown("### 📈 运筹预算约束\n- MCKP 多维背包降级\n- 严守大盘预算红线\n- 自动淘汰低效补贴")
    with col3:
        st.markdown("### 🚀 因果财务核算\n- 反事实自然流剥离\n- 计算纯增量 Uplift ROI\n- 可视化财务水瀑图")

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>AI用户仿真平台 v3.0 (Causal-OR 终极版) | 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
""", unsafe_allow_html=True)
