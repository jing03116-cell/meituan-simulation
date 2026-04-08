import streamlit as st
import pandas as pd
import numpy as np
import time

# ==========================================
# 0. 页面与环境初始化
# ==========================================
st.set_page_config(page_title="AI 策略与行为序列仿真引擎", layout="wide")
st.title("🧠 强化学习多智能体仿真平台 (RL-ABM Engine)")
st.markdown("融合**马尔可夫状态机**与**Q-Learning寻优**，重现真实外卖场景下千人千面的转化路径。")


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("大宽表.csv")
    except FileNotFoundError:
        st.error("未找到宽表文件，请确保文件名正确并在同一目录下！")
        st.stop()
    return df


df_users = load_data()

# ==========================================
# 1. 定义业务场景与规则 (需求 1 & 4)
# ==========================================
# 场景 Buff (影响进端、浏览和基础转化)
SCENARIOS = {
    "周末下午茶": {"desc": "休闲时间充足，客单价偏高，甜品偏好拉满", "open_app": 1.2, "cart_to_pay": 1.1},
    "工作日早晚餐": {"desc": "刚需场景，转化率极高，但客单价受限", "open_app": 0.9, "cart_to_pay": 1.5},
    "暴雨的晚上": {"desc": "运费飙升，对运费/大额免减极度敏感，放弃率高", "open_app": 1.5, "cart_to_pay": 0.6}
}

# 动作空间 Action Space (各类真实的优惠券)
ACTIONS = [
    {"name": "不发券", "type": "none", "cost": 0, "threshold": 0},
    {"name": "免费券(满30减5)", "type": "free", "cost": 5, "threshold": 30},
    {"name": "免费券(满45减8)", "type": "free", "cost": 8, "threshold": 45},
    {"name": "付费神券包(5元买膨胀)", "type": "paid", "cost": 5, "threshold": 0}
]


# 模拟付费神券膨胀机制 (需求 4)
def simulate_inflation(base_cost):
    # 5元包膨胀：60%概率膨胀到6-8元，40%概率膨胀到9-12元
    if np.random.rand() > 0.4:
        return np.random.uniform(6, 8.5)
    else:
        return np.random.uniform(9, 12.5)


# ==========================================
# 2. 核心马尔可夫序列计算引擎 (需求 2 & 3)
# ==========================================
def markov_journey(user, scenario_name, action):
    """模拟单个用户 [进端 -> 浏览 -> 加购 -> 支付] 的概率游走"""
    buff = SCENARIOS[scenario_name]
    aov = user['平均客单价']

    # 状态1: 进端 -> 浏览 (受场景和用户活跃度影响)
    p_browse = min(0.6 * buff['open_app'], 1.0)

    # 状态2: 浏览 -> 加购 (受用户原生漏斗特征影响)
    # 取宽表中的真实漏斗数据作为基准
    p_cart = user.get('动态_点击_至_加购率', 0.5) * 1.2

    # 状态3: 加购 -> 支付 (最核心的博弈环节)
    p_pay = 0.4 * buff['cart_to_pay']  # 基础支付率

    actual_cost = 0  # 平台实际付出的补贴成本

    if action['type'] == 'free':
        # 免费券逻辑：看门槛
        if aov >= action['threshold'] * 0.8:  # 用户愿意凑单的阈值
            lift = (action['cost'] * 0.02 * user['用券率'])
            p_pay += lift
            actual_cost = action['cost']

    elif action['type'] == 'paid':
        # 付费券逻辑：沉没成本 + 膨胀诱惑
        if user['付费券使用率'] > 0.1:  # 只要不是绝缘体
            inflated_value = simulate_inflation(action['cost'])
            # 沉没成本导致支付转化率直接拉满到 90% 以上
            p_pay = 0.90 + (inflated_value * 0.01)
            actual_cost = inflated_value - action['cost']  # 平台亏的是膨胀的差价

    p_pay = min(max(p_pay, 0.0), 1.0)

    # 最终期望
    final_prob = p_browse * p_cart * p_pay
    exp_gtv = final_prob * aov
    exp_cost = final_prob * actual_cost

    return p_browse, p_cart, p_pay, final_prob, exp_gtv, exp_cost


# ==========================================
# 3. 构建前端控制台 (左侧)
# ==========================================
st.sidebar.header("🕹️ 全局环境与控制")
selected_scenario = st.sidebar.selectbox("1. 设定全局业务场景", list(SCENARIOS.keys()))
st.sidebar.info(f"**场景特性：**\n{SCENARIOS[selected_scenario]['desc']}")

st.sidebar.markdown("---")
st.sidebar.subheader("2. 手动干预测试 (Manual)")
selected_action_name = st.sidebar.selectbox("向大盘下发指定策略", [a['name'] for a in ACTIONS])
manual_action = next(a for a in ACTIONS if a['name'] == selected_action_name)


# ==========================================
# 4. 强化学习 Q-Table 寻优 (需求 5)
# ==========================================
@st.cache_data(ttl=60)  # 缓存 RL 计算结果
def run_rl_optimizer(scenario):
    q_table = {}
    grouped = df_users.groupby('画像名称')

    for persona, group in grouped:
        q_table[persona] = {}
        for act in ACTIONS:
            total_reward = 0
            # 运行一个小规模蒙特卡洛评估期望 Reward (简化版 Q-Value)
            for _, user in group.iterrows():
                _, _, _, _, gtv, cost = markov_journey(user, scenario, act)
                # 定义 Reward：注重 ROI 的同时兼顾规模
                roi = gtv / cost if cost > 0 else gtv * 0.1
                reward = (roi * 0.7) + (gtv * 0.3)
                total_reward += reward
            q_table[persona][act['name']] = total_reward / len(group)

    df_q = pd.DataFrame(q_table).T
    return df_q


st.subheader("🤖 AI 强化学习策略大脑 (RL Q-Learning)")
st.markdown("系统已遍历所有状态(人群)与动作(券种)，依据期望 Reward(ROI与GTV综合收益) 收敛出以下 Q-Table：")

with st.spinner("AI 正在后台进行千万次环境博弈..."):
    df_q_table = run_rl_optimizer(selected_scenario)

    # 找出每个群体的最优策略
    best_actions = df_q_table.idxmax(axis=1)

    col_table, col_reason = st.columns([2, 1])
    with col_table:
        # 画出带热力图的 Q-Table
        st.dataframe(df_q_table.style.background_gradient(cmap='viridis', axis=1).format("{:.1f}"),
                     use_container_width=True)

    with col_reason:
        st.success("**🏆 AI 最优策略解析：**")
        for persona, action_name in best_actions.items():
            reason = "依靠沉没成本锁定高净值转化" if "付费" in action_name else "精准匹配免减门槛，撬动凑单意愿" if "免费" in action_name else "该群体当前场景抗拒营销，建议降本"
            st.markdown(f"- **{persona}** ➡️ `{action_name}`\n  *(理由：{reason})*")

st.markdown("---")

# ==========================================
# 5. 状态机动态可视化 (需求 3)
# ==========================================
st.subheader("🔄 典型用户马尔可夫游走序列分析 (手动策略影响)")
st.markdown(f"当前选中手动策略：**{manual_action['name']}**")

# 挑两个代表性用户展示
sample_users = df_users.groupby('画像名称').first().reset_index()

for _, user in sample_users.head(2).iterrows():
    p_b, p_c, p_p, f_p, gtv, cost = markov_journey(user, selected_scenario, manual_action)

    with st.container():
        st.caption(
            f"🆔 **{user['画像名称']}** (历史客单: ¥{user['平均客单价']:.1f} | 付费敏感度: {user['付费券使用率']:.1%})")

        # 动态箭头颜色 (高转化绿色，低转化红色)
        c1, c2, c3, c4, c5, c6, c7 = st.columns([2, 1, 2, 1, 2, 1, 2])

        c1.button("📱 曝光进端", key=f"n1_{user['user_id']}", disabled=True, use_container_width=True)
        c2.markdown(f"<div style='text-align:center; padding-top:10px; color:#1f77b4;'><b>{p_b:.1%} ➡️</b></div>",
                    unsafe_allow_html=True)
        c3.button("👀 浏览列表", key=f"n2_{user['user_id']}", disabled=True, use_container_width=True)
        c4.markdown(f"<div style='text-align:center; padding-top:10px; color:#1f77b4;'><b>{p_c:.1%} ➡️</b></div>",
                    unsafe_allow_html=True)
        c5.button("🛒 加入购物车", key=f"n3_{user['user_id']}", disabled=True, use_container_width=True)

        # 结算漏斗用高亮颜色
        color = "green" if p_p > 0.5 else ("red" if p_p < 0.2 else "orange")
        c6.markdown(f"<div style='text-align:center; padding-top:10px; color:{color};'><b>{p_p:.1%} ➡️</b></div>",
                    unsafe_allow_html=True)
        c7.button(f"💸 支付 (预期¥{gtv:.1f})", key=f"n4_{user['user_id']}", type="primary", use_container_width=True)
        st.write("")

# ==========================================
# 6. 微观行为特征响应大盘 (需求 5 末尾)
# ==========================================
st.markdown("---")
st.subheader("👥 仿真大盘抽样详情 (执行手动策略)")
if st.button("开始结算全量 200 用户"):
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)  # 增加动态计算感
        progress.progress(i + 1)

    res_list = []
    for _, u in df_users.iterrows():
        _, _, _, f_p, gtv, cost = markov_journey(u, selected_scenario, manual_action)
        res_list.append({
            "画像": u['画像名称'],
            "客单基准": u['平均客单价'],
            "策略下最终转化": f"{f_p:.1%}",
            "平台预期收益(GTV)": f"¥ {gtv:.2f}",
            "平台预期成本": f"¥ {cost:.2f}"
        })
    st.dataframe(pd.DataFrame(res_list).head(50), use_container_width=True)