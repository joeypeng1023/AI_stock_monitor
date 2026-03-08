import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import joblib
import torch
import torch.nn as nn
import os


# ================= 必须与训练时完全一致的 PyTorch 模型架构 =================
class StockLSTM(nn.Module):
    def __init__(self):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 只取最后一天输出的特征
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# ================= 页面配置与侧边栏 =================
st.set_page_config(page_title="看看你的股票 | AI 量化预测", page_icon="👀", layout="wide")

st.sidebar.header("🚨 价格预警设置")
alert_price = st.sidebar.number_input("目标触发价格 ($)", min_value=0.0, value=150.0, step=1.0)
alert_condition = st.sidebar.radio("触发条件", ["向上突破 (>=)", "向下跌破 (<=)"])
is_alert_active = st.sidebar.toggle("🟢 启用价格监控", value=False)
st.sidebar.divider()
st.sidebar.info("🧠 预测引擎已升级为：PyTorch LSTM 深度神经网络 (支持 Apple M 系列芯片硬件加速)")


# ================= 核心功能函数 =================
@st.cache_data(ttl=3600)
def get_daily_data(ticker):
    """获取数据并计算与训练时完全一致的特征"""
    try:
        # 获取近半年的数据即可，因为我们只需要最近 60 天的数据来做预测
        df = yf.Ticker(ticker).history(period="6mo")
        if df.empty:
            return None

        df.index = df.index.tz_localize(None)

        # 严格按照训练时的逻辑计算指标
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        df.dropna(inplace=True)
        return df
    except Exception as e:
        return None


def deep_learning_predict(df, symbol):
    """加载 PyTorch 模型进行明日趋势预测"""
    model_path = f"models/{symbol}_lstm_pytorch.pth"
    scaler_path = f'models/{symbol}_scaler.pkl'

    # 检查模型文件是否存在
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, f"⚠️ 未找到 {symbol} 的模型文件！请先在终端运行 train_lstm.py 训练该股票。"

    try:
        # 1. 加载 Scaler (数据解码器)
        scaler = joblib.load(scaler_path)

        # 2. 提取最近 60 天的 5 个特征
        feature_columns = ['Close', 'Volume', 'SMA_20', 'RSI_14', 'Log_Return']
        recent_60_days = df[feature_columns].tail(60).values

        if len(recent_60_days) < 60:
            return None, "⚠️ 数据不足 60 天，无法进行 LSTM 推理。"

        # 3. 归一化处理 (极其重要：必须用训练时的 scaler)
        scaled_data = scaler.transform(recent_60_days)

        # 4. 转换为 PyTorch Tensor (形状: 1个样本, 60个时间步, 5个特征)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        X_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)

        # 5. 加载模型并推理
        model = StockLSTM().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 切换到评估模式 (关闭 Dropout)

        with torch.no_grad():
            prediction = model(X_tensor)
            up_probability = prediction.item() * 100

        return up_probability, None

    except Exception as e:
        return None, f"预测出错: {str(e)}"


# ================= 主页面 UI =================
st.title("👀 看看你的股票")
st.caption(
    "⚠️ 声明：本系统基于 PyTorch 深度学习模型构建，预测结果仅供技术交流与参考，绝对不构成任何投资建议。股市有风险，入市需谨慎！")

# ---------------- 新增：自动扫描本地已训练模型 ----------------
# ---------------- 自动扫描本地已训练模型 ----------------
# 遍历当前目录，找出所有以 '_lstm_pytorch.pth' 结尾的文件，并提取股票代码
# ---------------- 自动扫描本地已训练模型 ----------------
# 遍历 models 文件夹，找出所有以 '_lstm_pytorch.pth' 结尾的文件
model_dir = "models"
if os.path.exists(model_dir):
    trained_tickers = [f.split('_')[0] for f in os.listdir(model_dir) if f.endswith('_lstm_pytorch.pth')]
else:
    trained_tickers = []

trained_tickers.sort() # 给代码排个序，看着更整齐

if not trained_tickers:
    st.warning("⚠️ 本地暂无已训练的模型，请先在终端运行 train_lstm.py 进行炼丹！")
    trained_tickers = ["请先训练模型"]


# ================= 新增：智能获取股票名称引擎 =================
@st.cache_data(show_spinner=False, ttl=86400)  # 魔法1：缓存一天，防止每次点开网页都去请求网络导致白屏卡顿
def get_stock_display_name(ticker_symbol):
    if ticker_symbol == "请先训练模型":
        return ticker_symbol
    try:
        # 魔法2：自动调用 yfinance 查户口
        info = yf.Ticker(ticker_symbol).info
        # 尝试获取简称，如果没有就抓全称，再没有就显示未知
        name = info.get('shortName', info.get('longName', '未知名称'))
        return f"{ticker_symbol} 🏢 {[name]}"
    except Exception:
        return f"{ticker_symbol} 🏢 未知名称"


# ============================================================

# ---------------- 终极防呆设计：下拉选择框 ----------------
symbol = st.selectbox(
    "🎯 请选择你要让 AI 预测的股票",
    options=trained_tickers,
    format_func=get_stock_display_name,  # <--- 魔法3：前后端分离显示
    help="列表会自动读取你本地已训练好的专属模型"
)

# 只有当用户选择了有效的股票代码时，按钮才允许点击
disable_btn = symbol == "请先训练模型"

if st.button("启动！"):
    with st.spinner(f"正在唤醒神经网络分析 {symbol} 走势..."):
        df = get_daily_data(symbol)

        if df is not None:
            # 调用 PyTorch 进行预测
            up_prob, error_msg = deep_learning_predict(df, symbol)

            latest_close = df.iloc[-1]['Close']
            current_rsi = df.iloc[-1]['RSI_14']
            current_sma = df.iloc[-1]['SMA_20']

            # ---------------- 1. 顶部预警 ----------------
            if is_alert_active:
                if alert_condition == "向上突破 (>=)" and latest_close >= alert_price:
                    st.error(f"🚨 警报触发！最新价 ${latest_close:.2f} 突破目标价 ${alert_price:.2f}！")
                elif alert_condition == "向下跌破 (<=)" and latest_close <= alert_price:
                    st.error(f"🚨 警报触发！最新价 ${latest_close:.2f} 跌破目标价 ${alert_price:.2f}！")
                else:
                    st.success(f"🛡️ 监控中：当前价格 ${latest_close:.2f}，安全。")

            st.divider()

            # ---------------- 2. 核心指标与深度学习预测结果 ----------------
            st.subheader("🧠 深度学习 (LSTM) 实时推理")

            if error_msg:
                st.warning(error_msg)
            else:
                # 第一行：只放 3 个基础市场指标，空间更宽裕
                col1, col2, col3 = st.columns(3)
                col1.metric("最新收盘价", f"${latest_close:.2f}")
                col2.metric("20日均线 (SMA)", f"${current_sma:.2f}")
                col3.metric("RSI (14日)", f"{current_rsi:.1f}", "超买 (>70) / 超卖 (<30)", delta_color="off")

                # 加一点垂直间距
                st.write("")

                # 第二行：AI 预测结果专属展示区
                # 加一点垂直间距
                st.write("")

                # ================= 终极视觉优化：巨型预测卡片 =================
                # 根据预测概率，动态生成卡片的背景色和文字颜色
                if up_prob >= 55:
                    bg_color = "#e6f4ea"  # 浅绿色背景
                    text_color = "#137333"  # 深绿色文字
                    trend_text = "🚀 强烈看涨"
                elif up_prob <= 45:
                    bg_color = "#fce8e6"  # 浅红色背景
                    text_color = "#c5221f"  # 深红色文字
                    trend_text = "📉 强烈看跌"
                else:
                    bg_color = "#e8f0fe"  # 浅蓝色背景
                    text_color = "#1967d2"  # 深蓝色文字
                    trend_text = "⚖️ 震荡不明"

                # 使用 HTML 和 CSS 渲染一个巨型卡片
                st.markdown(f"""
                                <div style="background-color: {bg_color}; padding: 30px; border-radius: 15px; text-align: center; border: 2px solid {text_color}; box-shadow: 2px 4px 10px rgba(0,0,0,0.1);">
                                    <h3 style="color: {text_color}; margin: 0; font-size: 24px;">AI 深度学习推理结论：{trend_text}</h3>
                                    <h1 style="color: {text_color}; font-size: 72px; margin: 15px 0; font-weight: 900;">{up_prob:.1f}%</h1>
                                    <p style="color: {text_color}; margin: 0; font-size: 20px; font-weight: bold;">( 明 日 上 涨 概 率 )</p>
                                </div>
                                """, unsafe_allow_html=True)
                # ================= 3. 历史回测胜率 (近期 100 个交易日) =================
                st.markdown("---")
                st.subheader("📊 模型近期胜率沙盘推演")

                with st.spinner("正在对过去 100 个交易日进行高频回测..."):
                    try:
                        import plotly.graph_objects as go
                        import numpy as np
                        import torch  # 确保引入了 torch

                        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

                        # 1. 抓取一年数据用于回测 (统一使用 history 接口防止多层索引报错)
                        df_bt = yf.Ticker(symbol).history(period="1y")
                        df_bt.index = df_bt.index.tz_localize(None)  # 清除时区信息防报错

                        # 2. 特征工程 (必须与训练时完全一致)
                        df_bt['Return'] = df_bt['Close'].pct_change()
                        df_bt['Log_Return'] = np.log(df_bt['Close'] / df_bt['Close'].shift(1))
                        df_bt['SMA_20'] = df_bt['Close'].rolling(window=20).mean()
                        delta = df_bt['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        df_bt['RSI_14'] = 100 - (100 / (1 + rs))
                        df_bt.dropna(inplace=True)  # 这一步会把刚算出来的空值（NaN）清理掉

                        # 生成真实的历史标签: 明日是否真的上涨了？(1 为涨, 0 为跌)
                        df_bt['Target'] = (df_bt['Return'].shift(-1) > 0).astype(int)
                        df_bt.dropna(inplace=True)  # 砍掉最后一天没实际结果的数据

                        # 引入 joblib 并加载专属的 scaler
                        import joblib

                        scaler = joblib.load(f"models/{symbol}_scaler.pkl")

                        # 提取特征并进行标准化
                        features_bt = df_bt[['Close', 'Volume', 'SMA_20', 'RSI_14', 'Log_Return']].values
                        scaled_features_bt = scaler.transform(features_bt)

                        # 3. 构造时间序列数据 (截取最后 100 个有效交易日)
                        X_bt, y_bt = [], []
                        lookback = 60
                        test_days = min(100, len(scaled_features_bt) - lookback)

                        for i in range(len(scaled_features_bt) - test_days, len(scaled_features_bt)):
                            X_bt.append(scaled_features_bt[i - lookback:i])
                            y_bt.append(df_bt['Target'].iloc[i])

                        X_bt = torch.tensor(np.array(X_bt), dtype=torch.float32).to(device)

                        # 4. 批量执行 AI 推理 (重新实例化并加载模型)
                        model_path_bt = f"models/{symbol}_lstm_pytorch.pth"
                        model_bt = StockLSTM().to(device)
                        model_bt.load_state_dict(torch.load(model_path_bt, map_location=device))
                        model_bt.eval()

                        with torch.no_grad():
                            # 注意这里用 model_bt 进行推理
                            outputs_bt = model_bt(X_bt).squeeze().cpu().numpy()

                        # 5. 统计胜率 (预测概率 > 0.5 且实际也上涨，或概率 < 0.5 且实际下跌)
                        preds_bt = (outputs_bt >= 0.5).astype(int)
                        correct_preds = np.sum(preds_bt == y_bt)
                        win_rate = (correct_preds / test_days) * 100

                        # 6. 渲染 Plotly 高级仪表盘 (完全适配深色模式)
                        # 动态核心色：胜率及格用亮绿，不及格用警示红
                        gauge_color = "#00E676" if win_rate >= 50 else "#FF5252"

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=win_rate,
                            # 🎯 痛点修复1：去掉黑体字，改为跟指针一致的动态高亮色，极其醒目
                            number={'suffix': "%", 'font': {'size': 50, 'color': gauge_color}},
                            title={'text': f"过去 {test_days} 个交易日预测胜率", 'font': {'size': 18}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                                'bar': {'color': gauge_color},  # 指针颜色联动
                                # 🎯 痛点修复2：去掉死白的底色，改为极淡的半透明色
                                'bgcolor': "rgba(128, 128, 128, 0.1)",
                                'borderwidth': 0,
                                'steps': [
                                    {'range': [0, 50], 'color': "rgba(255, 82, 82, 0.1)"},  # 红色半透明区间
                                    {'range': [50, 100], 'color': "rgba(0, 230, 118, 0.1)"}  # 绿色半透明区间
                                ],
                                'threshold': {
                                    'line': {'color': gauge_color, 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50}
                            }
                        ))

                        # 🎯 痛点修复3：将图表的画布背景彻底设为透明，完美融入系统深色模式
                        fig.update_layout(
                            height=350,
                            margin=dict(l=20, r=20, t=50, b=20),
                            paper_bgcolor="rgba(0,0,0,0)",  # 画布透明
                            plot_bgcolor="rgba(0,0,0,0)"  # 图表区域透明
                        )

                        # 强行开启 Streamlit 主题接管，它会自动把副标题等文字变成白色/深灰
                        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

                        # 给出智能解读提示
                        if win_rate > 55:
                            st.success(
                                f"💡 **AI 策略解读**: 该模型在近期表现优异，胜率显著跑赢 50% 的随机概率，当前预测结果具有**较高参考价值**。")
                        elif win_rate >= 50:
                            st.info(f"💡 **AI 策略解读**: 该模型在近期表现平稳，微弱跑赢抛硬币概率，请结合基本面综合判断。")
                        else:
                            st.warning(
                                f"⚠️ **AI 策略解读**: 该模型在近期胜率低于 50%（不及随机盲猜）。可能近期该股票行情逻辑发生突变，**强烈建议反向参考或空仓观望**！")

                    except Exception as e:
                        st.error(f"回测模块发生错误: {e}")

            # ---------------- 3. 可视化图表 ----------------
                    st.subheader(f"📊 {symbol} 价格走势（近100个交易日）")
                    plot_df = df.tail(100)

                    # 引入 make_subplots 制作带成交量的专业副图
                    from plotly.subplots import make_subplots

                    # 创建 2 行 1 列的画布，上图放 K 线(占70%)，下图放成交量(占30%)，共享 X 轴
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

                    # 1. 绘制专业级 K 线图 (实心荧光绿涨，实心警示红跌)
                    fig.add_trace(go.Candlestick(
                        x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                        low=plot_df['Low'], close=plot_df['Close'], name="K线",
                        increasing_line_color='#00E676', increasing_fillcolor='#00E676',
                        decreasing_line_color='#FF5252', decreasing_fillcolor='#FF5252'
                    ), row=1, col=1)

                    # 2. 绘制 20 日均线 (黄金色，增加质感)
                    fig.add_trace(go.Scatter(
                        x=plot_df.index, y=plot_df['SMA_20'],
                        line=dict(color='#FFCA28', width=2), name='SMA 20日均线',
                        hoverinfo='skip'  # 鼠标悬浮时不单独显示均线的提示框，避免干扰
                    ), row=1, col=1)

                    # 3. 绘制成交量柱状图 (根据当天的涨跌动态上色)
                    colors = ['#00E676' if row['Close'] >= row['Open'] else '#FF5252' for index, row in
                              plot_df.iterrows()]
                    fig.add_trace(go.Bar(
                        x=plot_df.index, y=plot_df['Volume'], name='成交量',
                        marker_color=colors, opacity=0.8
                    ), row=2, col=1)

                    # 4. 动态绘制价格预警线
                    if is_alert_active:
                        line_color = "#00E676" if alert_condition == "向上突破 (>=)" else "#FF5252"
                        fig.add_hline(y=alert_price, line_dash="dash", line_color=line_color,
                                      annotation_text=f"预警线: ${alert_price}", row=1, col=1)

                    # 5. 终极布局优化：剔除周末缺口、背景透明、加入十字准星
                    fig.update_layout(
                        xaxis_rangeslider_visible=False,  # 隐藏底部默认的丑陋滑块
                        height=650,
                        margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",  # 画布透明，完美融入深/浅色模式
                        plot_bgcolor="rgba(0,0,0,0)",  # 图表背景透明
                        showlegend=False,  # 隐藏占空间的图例
                        hovermode='x unified'  # 开启专业软件必备的“十字光标联动”效果！
                    )

                    # 6. 抹平非交易日产生的断层缺口，并加上极淡的参考网格
                    fig.update_xaxes(
                        rangebreaks=[dict(bounds=["sat", "mon"])],  # 隐藏周六到周一的空白
                        showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)'
                    )
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)')

                    # 渲染出图！
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("❌ 数据获取失败！请检查网络或股票代码。")