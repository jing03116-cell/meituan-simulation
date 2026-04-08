import streamlit as st
import pandas as pd
import numpy as np

# 1. 页面全局配置
st.set_page_config(page_title="美团智能补贴仿真沙盘", layout="wide")
st.title("🎯 美团单场景补贴策略 AI 仿真平台")
st.markdown("基于 ABM 多智能体引擎，实时读取 **[下午茶场景全息大宽表]** 进行商业推演。")


# 2. 读取数据 (使用缓存机制，让网页丝滑不卡顿)
@st.cache_data
def load_data():
    # 请确保这个文件名与你本地的大宽表文件名完全一致！
    df = pd.read_csv("大宽表.csv")
    return df


df_users = load_data()

# 3. 搭建左侧控制台 (UI)
st.sidebar.header("🕹️ 策略控制台")
strategy_type = st.sidebar.radio("分发策略选择：", ['全局大锅饭普发', 'AI 精准定向 (契合度>0.2)'])
free_val = st.sidebar.slider("免费满减券面额 (元)", min_value=0.0, max_value=15.0, value=5.0, step=1.0)
paid_val = st.sidebar.slider("付费膨胀神券面额 (元)", min_value=0.0, max_value=15.0, value=0.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.caption("引擎状态：200 虚拟智能体已就绪 🟢")


# 4. 核心仿真运算引擎 (Engine)
def run_simulation(df, strategy, free, paid):
    results = []
    base_prob = 0.05  # 自然转化率

    for _, user in df.iterrows():
        aov = user['平均客单价']
        sens_free = user['用券率']
        sens_paid = user['付费券使用率']
        # 计算下午茶场景契合度
        scene_affinity = (user['动态_下午茶活跃度'] + user.get('动态_饮品甜点偏好', 0)) / 2.0
        funnel_rate = user['动态_点击_至_加购率']

        # 定向过滤逻辑
        if strategy == 'AI 精准定向 (契合度>0.2)' and scene_affinity < 0.2:
            final_prob = base_prob
        else:
            theory_prob = base_prob + (free * 0.015 * sens_free) + (paid * 0.030 * sens_paid)
            final_prob = theory_prob * scene_affinity * funnel_rate

        final_prob = min(max(final_prob, 0.0), 1.0)

        gtv = final_prob * aov
        cost = final_prob * (free * sens_free + paid * sens_paid)

        results.append({
            'user_id': user['user_id'],
            '画像归属': user['画像名称'],
            '预测转化率': final_prob,
            '贡献GTV': gtv,
            '消耗成本': cost
        })
    return pd.DataFrame(results)


# 运行引擎
df_res = run_simulation(df_users, strategy_type, free_val, paid_val)

# 5. 渲染右侧数据大屏 (Dashboard)
total_gtv = df_res['贡献GTV'].sum()
total_cost = df_res['消耗成本'].sum()
roi = total_gtv / total_cost if total_cost > 0 else 0

st.subheader("📊 大盘 KPI 实时监控")
col1, col2, col3 = st.columns(3)
col1.metric("大盘总预期 GTV", f"¥ {total_gtv:,.1f}")
col2.metric("大盘总消耗成本", f"¥ {total_cost:,.1f}")
col3.metric("全局预期 ROI", f"{roi:.2f} x")

st.markdown("---")
col_chart, col_table = st.columns([1, 1])

with col_chart:
    st.subheader("📈 各画像群体 GTV 贡献对比")
    group_gtv = df_res.groupby('画像归属')['贡献GTV'].sum()
    st.bar_chart(group_gtv)

with col_table:
    st.subheader("👥 智能体个体响应明细 (Top 100)")
    # 美化表格展示
    show_df = df_res.head(100).style.format({
        '预测转化率': '{:.1%}', '贡献GTV': '¥{:.1f}', '消耗成本': '¥{:.1f}'
    })
    st.dataframe(show_df, use_container_width=True)