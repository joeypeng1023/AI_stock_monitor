import os
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

# ================= 配置区 =================
SYMBOL = '000426.SZ'
LOOKBACK_DAYS = 60
# =========================================

print(f"📥 正在获取 {SYMBOL} 历史数据...")
df = yf.Ticker(SYMBOL).history(period="5y")

if df.empty:
    print("❌ 数据获取失败")
    exit()

print("🛠️ 正在进行特征工程 (计算 RSI, SMA 等)...")
df['SMA_20'] = df['Close'].rolling(window=20).mean()

delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI_14'] = 100 - (100 / (1 + rs))
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df.dropna(inplace=True)

feature_columns = ['Close', 'Volume', 'SMA_20', 'RSI_14', 'Log_Return']
data = df[feature_columns].values

# 归一化并保存 Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
joblib.dump(scaler, f'{SYMBOL}_scaler.pkl')

X, y = [], []
close_idx = feature_columns.index('Close')

for i in range(LOOKBACK_DAYS, len(scaled_data) - 1):
    X.append(scaled_data[i - LOOKBACK_DAYS:i, :])
    if data[i + 1, close_idx] > data[i, close_idx]:
        y.append(1)
    else:
        y.append(0)

X, y = np.array(X), np.array(y)
train_size = int(len(X) * 0.8)

# 转换为 PyTorch 张量 (Tensor)
X_train = torch.tensor(X[:train_size], dtype=torch.float32)
y_train = torch.tensor(y[:train_size], dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X[train_size:], dtype=torch.float32)
y_test = torch.tensor(y[train_size:], dtype=torch.float32).unsqueeze(1)

# ================= 🚀 M4 芯片硬件加速 =================
# 自动检测并唤醒苹果 M 系列芯片的 MPS (Metal Performance Shaders) 引擎
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\n⚡ 硬件加速已启动！当前计算设备: {str(device).upper()} ⚡\n")

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


# ================= 搭建 PyTorch LSTM 架构 =================
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


model = StockLSTM().to(device)
criterion = nn.BCELoss()  # 二分类交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================= 开始炼丹 =================
print("🔥 开始训练神经网络...")
epochs = 50
batch_size = 32
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # 验证集测试
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)

        # 计算准确率
        predictions = (val_outputs >= 0.5).float()
        accuracy = (predictions == y_test).float().mean()

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch + 1:2d}/{epochs}], 训练误差: {loss.item():.4f}, 测试误差: {val_loss.item():.4f}, 准确率: {accuracy.item() * 100:.2f}%")

    # 确保 models 文件夹存在，如果不存在就自动创建
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, f"models/{symbol}_scaler.pkl")

    # 保存最好的模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"models/{symbol}_lstm_pytorch.pth")


print(f"🎉 {symbol} 模型训练完成，已安全存入 models 文件夹！")