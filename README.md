# 👀 看看你的股票 (CAN CAN NEED STOCK) | AI 全栈量化预测终端  
  
基于 PyTorch LSTM 的全栈量化分析与实时推理 Web 终端。本项目实现了一套端到端的 AI 股票预测闭环：集成了历史数据拉取、高阶量化因子构建、模型训练、动态沙盘回测验证，以及高度交互的专业级前端看板。
PyTorch LSTM-based full-stack quantitative analysis and real-time inference web terminal. This project implements an end-to-end set of AI stock prediction closed loops: integrating historical data pulling, high-order quantitative factor construction, model training, dynamic sand table backtesting and verification, and highly interactive professional-grade front-end Kanban.
  
---  
  
## ✨ 核心亮点 (Features)  
  
- **🧠 深度学习架构**: 基于 PyTorch 构建多层 LSTM 神经网络，完美解决时间序列的长期记忆问题。原生代码支持 Apple M系列芯片 (MPS) 及 CUDA 硬件级加速计算，实现光速矩阵推理。  

- **📊 动态沙盘回测**: 抛弃传统且具有欺骗性的“死循环”测试，利用滚动时间窗口对过去 100 个交易日进行即时推理，真实还原模型近期的推演胜率，严防“幸存者偏差”。  

- **📈 工业级可视化**: 抛弃简陋图表，集成 Plotly `make_subplots` 打造的专业级多图表联动（包含动态成交量副图、K线、十字光标全局联动），以及随胜率动态变色的荧光仪表盘 (Neon UI)。  

- **⚡ 沉浸式前端体验**: Streamlit 纯 Python 驱动，突破原生 UI 限制，深度适配操作系统的深色/浅色模式，利用内存级缓存 (`@st.cache_data`) 大幅降低 API 延迟与限流风险。  

- **🧠 Deep Learning Architecture**: Build a multi-layer LSTM neural network based on PyTorch, perfectly solving the long-term memory problem of time series. The native code supports Apple M-series chips (MPS) and CUDA hardware-level accelerated computing for light-speed matrix inference.

- **📊 Dynamic Sand Table Backtesting**: Abandon the traditional and deceptive "dead loop" testing and use a rolling time window to perform real-time inference on the past 100 trading days, truly restoring the model's recent deduction win rate and strictly preventing "survivor bias".

- **📈 Industrial-grade visualization**: Ditch the rudimentary charts and integrate Plotly's `make_subplots` with professional-grade multi-chart linkage (including dynamic volume subcharts, candlesticks, and crossbar global linkage), as well as fluorescent dashboards (Neon UI) that dynamically change color with win rate.

- **⚡ Immersive Front-End Experience**: Streamlit is powered by pure Python, breaking through the limitations of native UI, deeply adapting to the dark and light mode of the operating system, and utilizing memory-level caching ('@st.cache_data') to greatly reduce API latency and throttling risks.

## 🛠️ 技术栈 (Tech Stack)  
  
- **算法与深度学习引擎(Algorithms and deep learning engines)**: `PyTorch`, `LSTM`, `Scikit-Learn (Joblib)`, `Apple Metal Performance Shaders (MPS)`  

- **前端与可视化交互(The frontend interacts with visualizations)**: `Streamlit`, `Plotly Graph Objects`, `HTML/CSS 注入`  
  
---  
  
## 🚀 快速开始 (Quick Start)  
  
### 1. 环境安装  
推荐使用 Python 3.9+，克隆本仓库后，在终端执行以下命令安装依赖：  
`pip install -r requirements.txt`  
*(注：Mac M系列芯片用户推荐安装纯 CPU/MPS 版的 PyTorch 以优化内存占用。)*  
  
### 2. 模型训练 (模型初始化与特征构建)  
首次运行需要训练你的专属 AI 模型。运行以下脚本，系统会自动拉取股票数据、进行特征工程，并生成 `.pth` 模型权重文件和 `.pkl` 归一化缩放器：  
`python3 train_lstm.py`  

### 3. 启动量化终端   
启动可视化看板，见证 AI 预测与回测沙盘：  
`streamlit run dashboard.py`  
*(启动后，浏览器将自动打开 `http://localhost:8501`。)* 

### 1. Environment installation
It is recommended to use Python 3.9+, after cloning this repository, run the following command in the terminal to install dependencies:
`pip install -r requirements.txt` 

### 2. Model Training (Model Initialization and Feature Building)
The first run requires training your proprietary AI model. Run the following script and the system automatically pulls stock data, performs feature engineering, and generates a .pth model weight file and a .pkl normalization scaler:
`python3 train_lstm.py` 
  
### 3. Launch the Quantization Terminal
Launch the visual Kanban board to witness the AI prediction and backtesting sandbox: 
`streamlit run dashboard.py` 
*(After launching, the browser will automatically open `http://localhost:8501`.)*
  
---  
  
## 🤖 自动化运维 (Runned by OpenClaw)  
  
真正的量化系统不应该每天手动运行代码。本项目原生支持接入本地 AI Agent 
A true quantization system should not run code manually every day. This project natively supports connecting to local AI Agent
[OpenClaw](https://openclaw.ai/)。  
  
通过配置自然语言定时任务，系统将在**每周二至周六早晨 6:00**（美股周一至周五收盘后），自动拉取最新日线数据并重训模型，确保你每天早上看到的都是基于最新盘口数据生成的预测。  
By configuring natural language timing tasks, the system will automatically pull the latest daily data and retrain the model at **6:00 a.m. every Tuesday to Saturday morning** (after the close of U.S. stocks from Monday to Friday), ensuring that what you see every morning is a prediction based on the latest market data.
  
**在终端中运行以下指令以挂载自动化任务：**
**Run the following command in the terminal to mount the automation task:**  
  
`openclaw cron add --name "美股模型自动重训" \`    
  `--cron "0 6 * * 2-6" \`  
  `--tz "Asia/Shanghai" \`  
  `--message "请进入 /Your project absolute path/ 目录，静默运行 python3 train_lstm.py。完成后，请查阅本地生成的模型文件，并给我发送一份简短的晨报，告诉我今天有哪些股票模型完成了更新。"` 
   
*(**重要配置**：请务必将上方命令中的 `/你的项目绝对路径/` 替换为你本机存放 `train_lstm.py` 的真实文件夹路径。)*  
*(**Important configuration**: Be sure to replace '/your project absolute path/' in the command above with the real folder path where you store 'train_lstm.py' locally. )*
  
> 你可以随时在终端输入 `openclaw cron list` 来查看该任务的挂载状态。  
> You can always enter 'openclaw cron list' in the terminal to check the mount status of the task.
  
---  
  
## ⚠️ 免责声明 (Disclaimer)  
  
本项目提供的所有代码、模型结构及预测结果仅供**学术研究与技术交流**使用，绝对不构成任何投资建议。金融市场具有极高风险，AI 模型的历史回测胜率不代表未来真实收益。开发者不对任何基于本项目产生的交易亏损负责。入市有风险，投资需谨慎！  
All codes, model structures and prediction results provided by this project are for **academic research and technical exchange** only and absolutely do not constitute any investment advice. Financial markets are extremely risky, and the historical backtest win rate of AI models is not indicative of real future returns. The developer is not responsible for any trading losses incurred based on this project. Entering the market is risky, and investment needs to be cautious!
  
---  
*Created with ❤️ by: joeypeng1023 | 没事就敲敲代码的金融从业者*  
