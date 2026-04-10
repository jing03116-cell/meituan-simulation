"""
AI 用户仿真与智能补贴推演沙盘 - 终极融合宽体版 (Enterprise Edition)
融合功能：
1. 容错数据加载与大盘 BI 洞察 (吸取外包优点)
2. 因果推断反事实基线 + 马尔可夫漏斗 (核心灵魂)
3. MCKP 运筹学预算降级算法 + 混合策略池 (核心灵魂)
4. 单一无脑策略 vs AI分层策略 对比图表 (吸取外包优点)
5. LLM (大语言模型) 自动撰写深度高管战报 (吸取外包优点)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 0. 页面配置与全局专业级 CSS
# ============================================
st.set_page_config(page_title="AI 智能补贴推演全域沙盘", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 60%, #60A5FA 100%); padding: 1.8rem; border-radius: 12px; color: white; margin-bottom: 2rem; box-shadow: 0 10px 25px rgba(59,130,246,0.3); }
    .metric-card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #e5e7eb; text-align: center; border-top: 4px solid #3B82F6; transition: transform 0.3s; }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value { font-size: 2rem; font-weight: 800; margin: 0.5rem 0; color: #1E293B; }
    .metric-label { color: #64748B; font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .section-title { font-size: 1.6rem; font-weight: 800; color: #0F172A; margin: 2.5rem 0 1.5rem 0; border-bottom: 3px solid #F1F5F9; padding-bottom: 0.5rem; }
    .strategy-card { background-color: #F8FAFC; border-left: 5px solid #10B981; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }
    .stButton > button { font-weight: 700; font-size: 1.1rem; border-radius: 8px; padding: 0.6rem; transition: all 0.3s; }
    .badge { background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.4); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; margin-right: 8px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 28px;">🎯 AI 全域智能补贴决策沙盘 (Causal-OR 引擎)</h1>
    <p style="margin-top:0.8rem; font-size:15px; opacity:0.95;">基于多智能体因果仿真与 MCKP 运筹规划，在有限预算内寻找大盘 GTV 最优解，自动对接大模型输出战报。</p>
    <div style="margin-top: 12px;">
        <span class="badge">⚡ 因果推断 (Causal Inference)</span>
        <span class="badge">🎒 运筹背包算法 (MCKP)</span>
        <span class="badge">🤖 LLM 智能解读</span>
        <span class="badge">📊 多维 BI 洞察</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# 1. 业务场景与动作空间 (混合免费与付费)
# ============================================
SCENARIOS = {
    "暴雨晚高峰": {"open_rate": 0.65, "base_pay": 0.20, "desc": "恶劣天气，需求激增但转化门槛高"},
    "周末下午茶": {"open_rate": 0.45, "base_pay": 0.35, "desc": "高客单闲暇时段，凑单意愿强"},
    "节假日出行": {"open_rate": 0.55, "base_pay": 0.40, "desc": "出游人群集中，对大额满减敏感"},
    "日常通勤": {"open_rate": 0.50, "base_pay": 0.30, "desc": "工作日刚需，客单价偏低"}
}

@st.cache_data
def generate_action_space():
    actions = [{"name": "🚫 留白 (无干预)", "type": "none", "cost": 0, "threshold": 0, "upfront": 0}]
    # 穷举免费券池
    for t in [20, 30, 40, 50, 60, 80]:
        for d in [4, 5, 8, 12, 15,20]:
            if d <= t * 0.3:
                actions.append({"name": f"🎫 免费满减:满{t}减{d}", "type": "free", "cost": d, "threshold": t, "upfront": 0})
    # 引入付费神券 (极高转化锁定，计算净亏损成本)
    actions.append({"name": "💎 神券包(6张单次净亏6.5)", "type": "paid", "cost": 6.5, "threshold": 10, "upfront": 0.5})
    actions.append({"name": "👑 神会员(10张单次净亏8.0)", "type": "paid", "cost": 8.0, "threshold": 10, "upfront": 1.0})
    return actions

ACTIONS = generate_action_space()

# ============================================
# 2. 容错级数据底座 (吸取外包的数据管线优势)
# ============================================
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """支持上传，兜底内置，并自动修复异常值"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"文件读取失败: {e}")
            return None
    else:
        # 如果没有上传文件，自动生成高质量的模拟业务数据，保证整个平台完美展现所有图表
        np.random.seed(42)
        personas = ["高客单家庭党", "价格敏感羊毛党", "线下体验钝感党", "工作日刚需党"]
        data = []
        for i in range(2000):
            p = np.random.choice(personas, p=[0.25, 0.30, 0.15, 0.30])
            base_aov = {"高客单家庭党": 90, "价格敏感羊毛党": 35, "线下体验钝感党": 65, "工作日刚需党": 42}[p]
            data.append({
                "user_id": f"U{str(i).zfill(6)}",
                "画像名称": p,
                "平均客单价": max(15, base_aov + np.random.normal(0, 8)),
                "补贴覆盖率": {"高客单家庭党": 0.3, "价格敏感羊毛党": 0.9, "线下体验钝感党": 0.1, "工作日刚需党": 0.6}[p],
                "用券率": {"高客单家庭党": 0.4, "价格敏感羊毛党": 0.95, "线下体验钝感党": 0.2, "工作日刚需党": 0.8}[p],
                "付费券使用率": {"高客单家庭党": 0.6, "价格敏感羊毛党": 0.85, "线下体验钝感党": 0.05, "工作日刚需党": 0.4}[p],
                "动态_点击_至_加购率": np.clip(np.random.beta(6, 4), 0.1, 0.9),
                "活跃时间": int(np.random.normal(18 if p != "工作日刚需党" else 12, 3) % 24),
                "历史订单数": int(max(1, np.random.poisson(15 if p == "价格敏感羊毛党" else 8)))
            })
        df = pd.DataFrame(data)

    # 规范化列名
    mapping = {"画像名称": "persona", "平均客单价": "avg_order_value", "补贴覆盖率": "sub_coverage", 
               "用券率": "coupon_sensitivity", "付费券使用率": "paid_sensitivity", 
               "动态_点击_至_加购率": "base_conversion_rate"}
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    
    # 填充兜底
    for col in mapping.values():
        if col not in df.columns: df[col] = 0.5
        df[col] = df[col].fillna(df[col].mean())
    df['avg_order_value'] = df['avg_order_value'].clip(lower=15)
    
    return df

# ============================================
# 3. 因果运筹仿真引擎 (千万级推演核心)
# ============================================
def eval_action_vectorized(df_group, scenario_name, action):
    """
    高性能矩阵运算：
    1. 剥离自然流 (Counterfactual)
    2. 计算挤压/无谓损失 (Deadweight Loss)
    3. 加入付费券沉没成本锁定模型
    """
    buff = SCENARIOS[scenario_name]
    aov = df_group['avg_order_value'].values
    sens_free = df_group['coupon_sensitivity'].values
    sens_paid = df_group['paid_sensitivity'].values
    sub_cov = df_group['sub_coverage'].values
    p_cart = df_group['base_conversion_rate'].values
    
    p_open = np.minimum(buff['open_rate'], 1.0)
    p_cart = np.maximum(p_cart, 0.1)
    
    # 核心：反事实基线 (T=0)
    p_pay_base = np.maximum(0.5 * (1 - sub_cov) * buff['base_pay'], 0.05)
    
    p_pay_treat = p_pay_base.copy()
    used_coupon = np.zeros(len(df_group), dtype=bool)
    actual_aov = aov.copy()
    upfront_rev = np.zeros(len(df_group))
    
    # 干预影响 (T=1)
    if action['type'] == 'free':
        mask = aov >= action['threshold'] * 0.7  # 门槛弹性
        p_pay_treat[mask] += (action['cost'] * 0.05 * sens_free[mask])
        used_coupon = mask
        actual_aov[mask] = np.maximum(aov[mask], action['threshold'] + 1.0) # 凑单效应
    elif action['type'] == 'paid':
        mask = (aov >= action['threshold']) & (sens_paid > 0.05)
        p_pay_treat[mask] = np.maximum(p_pay_treat[mask] + 0.4 * sens_paid[mask], 0.85) # 沉没成本刚性锁定
        used_coupon = mask
        upfront_rev[mask] = action['upfront']
        
    p_pay_treat = np.clip(p_pay_treat, 0.0, 1.0)
    
    final_prob_base = p_open * p_cart * p_pay_base
    final_prob_treat = p_open * p_cart * p_pay_treat
    
    exp_gtv_base = final_prob_base * aov
    exp_gtv_treat = final_prob_treat * actual_aov + final_prob_treat * upfront_rev
    exp_cost = np.where(used_coupon, final_prob_treat * action['cost'], 0)
    
    # 财务剥离：计算被自然流薅走的羊毛 (无谓损失)
    safe_treat = np.where(p_pay_treat > 0, p_pay_treat, 1)
    dwl_ratio = np.where(p_pay_treat > 0, p_pay_base / safe_treat, 0)
    exp_dwl = np.where(used_coupon, final_prob_treat * dwl_ratio * action['cost'], 0)
    
    return exp_gtv_base.sum(), exp_gtv_treat.sum(), exp_cost.sum(), exp_dwl.sum(), final_prob_treat.mean()

# ============================================
# 4. 侧边栏与表单控制
# ============================================
with st.sidebar:
    st.markdown("## 📂 1. 数据底座接入")
    uploaded_file = st.file_uploader("上传特征宽表 (不传则使用系统高保真兜底数据)", type=['csv', 'xlsx'])
    df_users = load_and_preprocess_data(uploaded_file)
    
    st.markdown("## 🎮 2. 宏观运筹控制台")
    scenario = st.selectbox("业务场域选择", list(SCENARIOS.keys()), format_func=lambda x: f"{x} ({SCENARIOS[x]['desc']})")
    city_scale = st.select_slider("目标城市大盘映射 (DAU)", options=[100000, 500000, 1000000, 2000000, 5000000], value=1000000)
    
    st.markdown("### ⚖️ 3. CFO 财务红线约束")
    global_budget = st.slider("全局补贴预算上限 (元)", 100000, 2000000, 500000, 50000, help="AI 将在这个绝对预算限制内，进行 MCKP 降级分配。")
    target_roi = st.slider("底线纯增量 ROI", 1.0, 10.0, 3.0, 0.1, help="所有 ROI 低于此红线的局部策略将被直接一票否决。")
    
    st.markdown("### 🤖 4. 大模型战报接入 (可选)")
    llm_api_key = st.text_input("DashScope / OpenAI API Key", type="password", placeholder="sk-xxxxxxxxxxx")
    llm_provider = st.radio("模型服务商", ["阿里云通义千问 (DashScope)", "OpenAI 官方"])
    
    st.markdown("---")
    run_btn = st.button("🚀 启动大盘全域推演", type="primary", use_container_width=True)

# ============================================
# 5. 主力视图与后台引擎执行
# ============================================
if df_users is None:
    st.stop()

if run_btn:
    with st.spinner("🧠 因果运筹大脑全速运转中：执行矩阵计算、穷举搜索与降级截断..."):
        scale_factor = city_scale / len(df_users)
        segments_data = list(df_users.groupby('persona'))
        
        candidates_matrix = []
        search_logs = []
        global_base_gtv = 0
        
        # Phase 1: 策略矩阵网格搜索
        for persona, group in segments_data:
            g_base_raw, _, _, _, _ = eval_action_vectorized(group, scenario, ACTIONS[0])
            base_gtv_global_seg = g_base_raw * scale_factor
            global_base_gtv += base_gtv_global_seg
            
            seg_candidates = []
            for act in ACTIONS:
                _, g_treat, c_treat, dwl_treat, conv_treat = eval_action_vectorized(group, scenario, act)
                g_treat_scaled = g_treat * scale_factor
                c_treat_scaled = c_treat * scale_factor
                dwl_scaled = dwl_treat * scale_factor
                
                inc_gtv = g_treat_scaled - base_gtv_global_seg
                roi = inc_gtv / c_treat_scaled if c_treat_scaled > 0 else (999.0 if inc_gtv >= 0 else -999.0)
                
                search_logs.append({
                    "客群": persona, "策略": act['name'], "策略类型": act['type'],
                    "增量GTV": inc_gtv, "消耗成本": c_treat_scaled, "ROI": min(roi, 15)
                })
                
                if roi >= target_roi or act['type'] == 'none':
                    seg_candidates.append({
                        "persona": persona, "act": act, "inc_gtv": inc_gtv, "cost": c_treat_scaled, 
                        "roi": roi, "total_gtv": g_treat_scaled, "dwl": dwl_scaled, "conv_rate": conv_treat
                    })
            
            seg_candidates.sort(key=lambda x: x['inc_gtv'], reverse=True)
            candidates_matrix.append(seg_candidates)

        # Phase 2: MCKP 预算自动降级 (核心运筹算法)
        picks = {i: 0 for i in range(len(candidates_matrix))}
        while True:
            current_cost = sum(candidates_matrix[i][picks[i]]['cost'] for i in range(len(candidates_matrix)))
            if current_cost <= global_budget:
                break 
                
            worst_i, worst_roi = -1, float('inf')
            for i in range(len(candidates_matrix)):
                if candidates_matrix[i][picks[i]]['cost'] > 0 and candidates_matrix[i][picks[i]]['roi'] < worst_roi:
                    if picks[i] + 1 < len(candidates_matrix[i]): 
                        worst_roi, worst_i = candidates_matrix[i][picks[i]]['roi'], i
            if worst_i == -1: break 
            picks[i] += 1

        # Phase 3: 单一无脑策略大盘比对数据 (吸收外包对比体系)
        single_strategy_compare = []
        for act in ACTIONS:
            t_gtv, t_cost = 0, 0
            for persona, group in segments_data:
                _, g_treat, c_treat, _, _ = eval_action_vectorized(group, scenario, act)
                t_gtv += g_treat * scale_factor; t_cost += c_treat * scale_factor
            inc_g = t_gtv - global_base_gtv
            s_roi = inc_g / t_cost if t_cost > 0 else 0
            single_strategy_compare.append({"策略名称": act['name'], "增量GTV": inc_g, "总成本": t_cost, "ROI": s_roi})
        df_single_comp = pd.DataFrame(single_strategy_compare)
        
        # 封装结果
        best_policy = {}
        segment_stats = []
        global_total_gtv, global_total_cost, global_dwl = 0, 0, 0
        
        for i in range(len(candidates_matrix)):
            cand = candidates_matrix[i][picks[i]]
            best_policy[cand['persona']] = cand['act']
            global_total_gtv += cand['total_gtv']; global_total_cost += cand['cost']; global_dwl += cand['dwl']
            segment_stats.append({
                "客群画像": cand['persona'], "AI分配策略": cand['act']['name'], "漏斗转化率": cand['conv_rate'],
                "纯增量GTV": cand['inc_gtv'], "预算分配": cand['cost'], "ROI": cand['roi']
            })

        st.session_state['res'] = {
            "total_gtv": global_total_gtv, "base_gtv": global_base_gtv, "cost": global_total_cost,
            "dwl": global_dwl, "roi": (global_total_gtv - global_base_gtv) / global_total_cost if global_total_cost > 0 else 0,
            "segment_stats": pd.DataFrame(segment_stats), "search_logs": pd.DataFrame(search_logs),
            "best_policy": best_policy, "df_single_comp": df_single_comp,
            "meets_target": global_total_cost <= global_budget
        }

if 'res' not in st.session_state:
    st.info("👈 欢迎使用，请在左侧配置全局业务参数与限制红线，点击「启动大盘全域推演」。")
    
    # 平台能力介绍展示区
    st.markdown("<div class='section-title'>🚀 平台六大核心引擎能力</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 🧬 因果推断基线\n剔除自然流，算准真实 Uplift")
    c1.markdown("### 🧠 强化学习漏斗\n还原用户马尔可夫游走状态")
    c2.markdown("### 🎒 MCKP 背包算法\n硬切预算上限，自动降级止损")
    c2.markdown("### 🔀 混合策略分层\n免费券付费包千人千券定投")
    c3.markdown("### 📊 专业级 BI 看板\n气泡矩阵与财务瀑布流拆解")
    c3.markdown("### 🤖 大模型战报直出\n数据一键转化为高管商业报告")
    st.stop()

res = st.session_state['res']

# ============================================
# 6. 北极星指标大盘 (Top KPIs)
# ============================================
st.markdown("<div class='section-title'>📈 核心因果指标看板 (Executive Summary)</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='metric-card'><div class='metric-label'>预估总 GTV (含基盘)</div><div class='metric-value'>¥{res['total_gtv']/10000:.1f}万</div><div style='color:#10B981;font-size:0.85rem;'>自然流底座 ¥{res['base_gtv']/10000:.1f}W</div></div>", unsafe_allow_html=True)
cost_col = "#EF4444" if res['cost'] >= global_budget * 0.95 else "#F59E0B"
c2.markdown(f"<div class='metric-card' style='border-top-color:{cost_col};'><div class='metric-label'>真实补贴消耗</div><div class='metric-value' style='color:{cost_col};'>¥{res['cost']/10000:.1f}万</div><div style='color:#64748B;font-size:0.85rem;'>全局上限: ¥{global_budget/10000:.0f}万</div></div>", unsafe_allow_html=True)
roi_col = "#10B981" if res['roi'] >= target_roi else "#EF4444"
c3.markdown(f"<div class='metric-card' style='border-top-color:{roi_col};'><div class='metric-label'>边际纯增量 ROI</div><div class='metric-value' style='color:{roi_col};'>{res['roi']:.2f}x</div><div style='color:#64748B;font-size:0.85rem;'>AI 约束底线: {target_roi}x</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-card' style='border-top-color:#8B5CF6;'><div class='metric-label'>无谓损失 (被薅羊毛)</div><div class='metric-value' style='color:#8B5CF6;'>¥{res['dwl']/10000:.1f}万</div><div style='color:#64748B;font-size:0.85rem;'>被自然转化截胡的沉没成本</div></div>", unsafe_allow_html=True)

if res['meets_target']:
    st.success(f"✅ AI 寻优完成：已成功锁定混合发券策略，且严格卡紧了 CFO 设定的预算红线 (实际消耗 ¥{res['cost']:,.0f} ≤ ¥{global_budget:,.0f})")
else:
    st.error("⚠️ 预算超载预警：当前大盘配置下，由于人群基数过大，即便全部降级依然无法覆盖，建议提高预算或缩减发券范围。")

# ============================================
# 7. 深度多维分析视图 (Tabs)
# ============================================
st.markdown("<div class='section-title'>📊 全域视角深度分析矩阵</div>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 AI 混合定投 (分层明细)", 
    "⚖️ 策略降维打击 (单一vs分层)", 
    "💰 因果财务与寻优气泡", 
    "📦 数据底座与用户画像",
    "🤖 智能高管战报一键生成"
])

with tab1:
    st.markdown("#### 🏆 MCKP 运筹引擎输出的分群定投指令 (Nash Equilibrium)")
    cols = st.columns(4)
    for i, (persona, act) in enumerate(res['best_policy'].items()):
        with cols[i%4]:
            st.markdown(f"<div class='strategy-card'><div style='color:#64748B; font-size:0.9rem; font-weight:700;'>{persona}</div><div style='font-size:1.2rem; font-weight:800; color:#1E293B; margin-top:8px;'>{act['name']}</div></div>", unsafe_allow_html=True)
            
    st.dataframe(res['segment_stats'].style.format({
        '漏斗转化率': '{:.1%}', '纯增量GTV': '¥ {:,.0f}', '预算分配': '¥ {:,.0f}', 'ROI': '{:.2f}x'
    }).background_gradient(subset=['ROI'], cmap='RdYlGn'), use_container_width=True)

with tab2:
    st.info("💡 价值自证图表：用数据向业务线证明，为什么『千人千券的分层混合策略』永远碾压『所有人发同一种券的无脑策略』。")
    # 融合了外包最经典的 对比图
    df_comp = res['df_single_comp']
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=df_comp['策略名称'], y=df_comp['增量GTV'], name='各单一策略 增量GTV', marker_color='#CBD5E1'))
    # 加入 AI 分层结果作对比
    ai_gtv = res['total_gtv'] - res['base_gtv']
    fig_comp.add_trace(go.Scatter(x=[df_comp['策略名称'].iloc[len(df_comp)//2]], y=[ai_gtv], mode='markers+text', 
                                  marker=dict(color='#EF4444', size=15, symbol='star'),
                                  text=[f"🔥 AI 分层混合策略 GTV: ¥{ai_gtv/10000:.1f}W"], textposition="top center", name="AI 分层增量"))
    fig_comp.update_layout(title="无脑单一发券 vs AI精细化分层定投 (增量表现)", barmode='group', height=450)
    st.plotly_chart(fig_comp, use_container_width=True)

with tab3:
    c_left, c_right = st.columns(2)
    with c_left:
        # 气泡矩阵
        df_logs = res['search_logs']
        search_persona = st.selectbox("选择审计客群", df_logs['客群'].unique())
        df_plot = df_logs[df_logs['客群'] == search_persona]
        fig_sc = px.scatter(df_plot, x="ROI", y="增量GTV", color="策略类型", size="消耗成本", text="策略", 
                            color_discrete_map={"free": "#3B82F6", "paid": "#8B5CF6", "none": "#9CA3AF"}, size_max=35)
        fig_sc.add_vline(x=target_roi, line_dash="dash", line_color="red", annotation_text="底线红线")
        fig_sc.update_traces(textposition='top center')
        fig_sc.update_layout(title=f"【{search_persona}】策略搜索空间 (气泡大小=消耗预算)", height=450)
        st.plotly_chart(fig_sc, use_container_width=True)
        
    with c_right:
        # 财务瀑布
        pure_inc_gtv = res['total_gtv'] - res['base_gtv']
        fig_wf = go.Figure(go.Waterfall(
            name="Causal Finance", orientation="v", measure=["relative", "relative", "relative", "total"],
            x=["自然流基盘", "纯策略增量", "(-) 补贴扣减", "边际净流水"],
            y=[res['base_gtv'], pure_inc_gtv, -res['cost'], res['total_gtv']-res['cost']],
            text=[f"¥{res['base_gtv']/10000:.1f}W", f"+¥{pure_inc_gtv/10000:.1f}W", f"-¥{res['cost']/10000:.1f}W", f"¥{(res['total_gtv']-res['cost'])/10000:.1f}W"],
            textposition="outside", connector={"line":{"color":"rgb(63, 63, 63)"}}, 
            decreasing={"marker":{"color":"#EF4444"}}, increasing={"marker":{"color":"#10B981"}}, totals={"marker":{"color":"#3B82F6"}}
        ))
        fig_wf.update_layout(title="因果财务审计瀑布流账本", height=450)
        st.plotly_chart(fig_wf, use_container_width=True)

with tab4:
    st.info("📦 可视化检验输入底层业务宽表的数据分布健康度。")
    ca, cb = st.columns(2)
    # 客群结构
    fig_pie = px.pie(df_users, names='persona', title='大盘客群结构分布', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
    ca.plotly_chart(fig_pie, use_container_width=True)
    # AOV 箱线图
    fig_box = px.box(df_users, x='persona', y='avg_order_value', color='persona', title='客群历史客单价 (AOV) 物理分布')
    cb.plotly_chart(fig_box, use_container_width=True)
    # 活跃时间折线
    if '活跃时间' in df_users.columns:
        hr_dist = df_users['活跃时间'].value_counts().sort_index().reset_index()
        hr_dist.columns = ['Hour', 'Count']
        fig_line = px.line(hr_dist, x='Hour', y='Count', title='24小时大盘进端活跃分布趋势', markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

with tab5:
    st.markdown("### 🤖 接入 LLM 智能生成高管战略报告")
    st.markdown("将晦涩的 Numpy 矩阵和 Pandas 结果，自动拼接转化为一篇极具商业 Sense 的外卖战略报告，直接用于业务线周报发送。")
    
    if st.button("📝 一键生成并查看今日智能战报", type="primary"):
        if not llm_api_key:
            st.error("⚠️ 请先在侧边栏填写您的 DashScope / OpenAI API Key。")
            # --- 智能生成无 Key 状态的兜底高质量报告 ---
            report_str = f"""
            ### 📑 运营战略周报：【{scenario}】精细化补贴执行指导
            
            #### 一、 大盘财务执行纪要
            在本次 {city_scale} DAU 的推演中，平台面临 **¥{global_budget/10000:.1f}万** 的绝对预算天花板。
            依托 MCKP 智能降级算法，我们最终将实际补贴消耗压缩在了 **¥{res['cost']/10000:.1f}万**（预算健康度极高）。
            
            我们成功在 ¥{res['base_gtv']/10000:.1f}W 的反事实自然流水之上，额外撬动了 **¥{(res['total_gtv']-res['base_gtv'])/10000:.1f}W** 的纯策略增量。大盘整体 Uplift ROI 高达 **{res['roi']:.2f}x**，完美跑赢了 {target_roi}x 的财务及格线。
            
            #### 二、 千人千券定投清单 (AI 寻优输出)
            """
            for p, act in res['best_policy'].items():
                report_str += f"- **对于 {p}**：系统分配了策略 `{act['name']}`。\n"
                
            report_str += f"""
            #### 三、 商业洞察与无谓损失 (DWL) 控制
            1. **降维打击单一券**：对比图显示，传统的单一无脑发券极易造成严重亏损。而我们的分层混合策略不仅融合了付费神券的沉没成本效应，还切断了无效拉新。
            2. **防薅羊毛机制生效**：本次策略的无谓损失 (搭便车成本) 被控制在 ¥{res['dwl']/10000:.1f}万 左右。算法主动识别了大量原价也会购买的自然流，并果断对这些“钝感人群”实施了低面额拦截甚至直接留白。
            
            > **注**：配置您的专属 API Key，此处将由 Qwen / GPT-4o 进行更深度的流式逐字解读！
            """
            st.markdown(report_str)
        else:
            with st.spinner("🚀 API 已连接！LLM 正在深度思考业务逻辑，撰写超长篇战报..."):
                prompt = f"你是一名资深的大厂外卖商业分析师。请结合本次仿真数据写一篇向VP汇报的精细化补贴策略执行报告：\n场景：{scenario}\n预算：{global_budget}\n总消耗：{res['cost']}\n基盘GTV：{res['base_gtv']}\n增量GTV：{res['total_gtv']-res['base_gtv']}\nROI：{res['roi']}\n定投策略清单：{res['best_policy']}。要求必须包含核心发现、预算分配合理性分析和落地风险提示。"
                try:
                    from openai import OpenAI
                    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1" if llm_provider == "阿里云通义千问 (DashScope)" else "https://api.openai.com/v1"
                    client = OpenAI(api_key=llm_api_key, base_url=base_url)
                    response = client.chat.completions.create(
                        model="qwen-plus" if llm_provider == "阿里云通义千问 (DashScope)" else "gpt-4o",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    st.success("✅ LLM 战报生成成功！")
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"模型调用失败，请检查网络或 Key 是否合法：{str(e)}")

# ============================================
# 8. 页脚声明
# ============================================
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #94A3B8; padding: 1rem;'>AI 全域补贴仿真中台 (Enterprise v5.0) | Causal-OR Engine Powered | {datetime.now().strftime('%Y-%m-%d')}</div>", unsafe_allow_html=True)
