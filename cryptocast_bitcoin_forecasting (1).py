"""
╔══════════════════════════════════════════════════════════════╗
║   CryptoCast: Multi-Horizon Bitcoin Price Forecasting        ║
║   Models: 1D-CNN, RNN, LSTM, Transformer (PyTorch)          ║
║   Horizons: 1-Day, 3-Day, 7-Day                              ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────
# STEP 1: IMPORT LIBRARIES
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Libraries loaded | Device: {DEVICE}")


# ─────────────────────────────────────────────
# STEP 2: LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("Bitcoin Historical Data (1) (2).csv")
print(f"\n📊 Shape: {df.shape}")
print(df.head(3))


# ─────────────────────────────────────────────
# STEP 3: PREPROCESS DATA
# ─────────────────────────────────────────────
def preprocess(df):
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data = data.sort_values('Date').reset_index(drop=True)

    for col in ['Price', 'Open', 'High', 'Low']:
        data[col] = data[col].astype(str).str.replace(',', '').astype(float)

    def fix_vol(v):
        if isinstance(v, str):
            v = v.strip()
            if 'K' in v: return float(v.replace('K','')) * 1e3
            if 'M' in v: return float(v.replace('M','')) * 1e6
            if 'B' in v: return float(v.replace('B','')) * 1e9
            try: return float(v)
            except: return np.nan
        return v

    data['Vol.'] = data['Vol.'].apply(fix_vol)
    data['Change %'] = data['Change %'].astype(str).str.replace('%','').astype(float)

    data = data.ffill().bfill()
    print(f"\n✅ Preprocessed | {data['Date'].min().date()} → {data['Date'].max().date()}")
    print(f"   Rows: {len(data)}")
    return data

data = preprocess(df)


# ─────────────────────────────────────────────
# STEP 4: EDA PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
fig.suptitle('Bitcoin Historical Data — EDA', fontsize=14, fontweight='bold')

axes[0,0].plot(data['Date'], data['Price'], color='orange', lw=1.2)
axes[0,0].set_title('Closing Price Over Time')
axes[0,0].set_ylabel('Price (USD)')
axes[0,0].grid(alpha=0.3)

colors_bar = ['green' if float(str(x).replace('%','')) >= 0 else 'red' for x in data['Change %']]
axes[0,1].bar(data['Date'], data['Change %'], color=colors_bar, alpha=0.7, width=1)
axes[0,1].set_title('Daily Change %')
axes[0,1].axhline(0, color='black', lw=0.8)
axes[0,1].grid(alpha=0.3)

axes[1,0].fill_between(data['Date'], data['Vol.'], alpha=0.6, color='steelblue')
axes[1,0].set_title('Trading Volume')
axes[1,0].grid(alpha=0.3)

axes[1,1].hist(data['Price'], bins=50, color='orange', edgecolor='black', alpha=0.7)
axes[1,1].set_title('Price Distribution')
axes[1,1].set_xlabel('Price (USD)')
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: eda_plots.png")


# ─────────────────────────────────────────────
# STEP 5: SCALE & CREATE SEQUENCES
# ─────────────────────────────────────────────
FEATURES = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
SEQ_LEN  = 60
HORIZONS = [1, 3, 7]

scaler       = MinMaxScaler()
price_scaler = MinMaxScaler()

scaled = scaler.fit_transform(data[FEATURES])
price_scaler.fit(data[['Price']])

def make_sequences(scaled, seq_len, horizon, price_idx=0):
    X, y = [], []
    for i in range(seq_len, len(scaled) - horizon + 1):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i+horizon-1, price_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def time_split(X, y, ratio=0.2):
    s = int(len(X)*(1-ratio))
    return X[:s], X[s:], y[:s], y[s:]

datasets = {}
for h in HORIZONS:
    X, y = make_sequences(scaled, SEQ_LEN, h)
    Xtr, Xte, ytr, yte = time_split(X, y)
    datasets[h] = dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)
    print(f"   {h}D → Train {Xtr.shape}, Test {Xte.shape}")

N_FEAT = scaled.shape[1]
print(f"\n✅ Sequences ready | Input: ({SEQ_LEN}, {N_FEAT})")


# ─────────────────────────────────────────────
# STEP 6: MODEL DEFINITIONS (PyTorch)
# ─────────────────────────────────────────────

class CNNModel(nn.Module):
    def __init__(self, seq_len, n_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),    nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(128*8, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):          # x: (B, T, F)
        return self.net(x.permute(0,2,1)).squeeze(-1)


class RNNModel(nn.Module):
    def __init__(self, seq_len, n_feat):
        super().__init__()
        self.rnn = nn.RNN(n_feat, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc  = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:,-1,:]).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, seq_len, n_feat):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32,1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:]).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, seq_len, n_feat):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, 64)
        encoder_layer   = nn.TransformerEncoderLayer(d_model=64, nhead=4,
                            dim_feedforward=128, dropout=0.1, batch_first=True)
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc         = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32,1))
    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1)).squeeze(-1)


print("✅ Model architectures defined!")


# ─────────────────────────────────────────────
# STEP 7: TRAIN FUNCTION
# ─────────────────────────────────────────────
def train_model(model, Xtr, ytr, epochs=60, batch=32, lr=1e-3, patience=10):
    model.to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset   = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    loader    = DataLoader(dataset, batch_size=batch, shuffle=False)

    val_split = int(len(Xtr)*0.85)
    Xv = torch.tensor(Xtr[val_split:]).to(DEVICE)
    yv = torch.tensor(ytr[val_split:]).to(DEVICE)
    Xt = torch.tensor(Xtr[:val_split]).to(DEVICE)
    yt = torch.tensor(ytr[:val_split]).to(DEVICE)

    train_losses, val_losses = [], []
    best_val, patience_cnt, best_state = 999, 0, None

    for ep in range(epochs):
        model.train()
        ep_loss = 0
        for xb, yb in DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=False):
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(Xv), yv).item()
        train_losses.append(ep_loss/len(Xt)*batch)
        val_losses.append(vl)

        if vl < best_val:
            best_val, patience_cnt = vl, 0
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_state)
    return train_losses, val_losses


# ─────────────────────────────────────────────
# STEP 8: TRAIN ALL MODELS × ALL HORIZONS
# ─────────────────────────────────────────────
MODEL_CLASSES = {'1D-CNN': CNNModel, 'RNN': RNNModel,
                 'LSTM': LSTMModel, 'Transformer': TransformerModel}

all_results   = {}
all_histories = {}

print("\n🚀 Training starts...\n")

for mname, MClass in MODEL_CLASSES.items():
    all_results[mname]   = {}
    all_histories[mname] = {}

    for h in HORIZONS:
        print(f"  ⏳ {mname} | {h}D ...", end=' ', flush=True)
        d = datasets[h]

        model = MClass(SEQ_LEN, N_FEAT)
        tr_loss, val_loss = train_model(model, d['Xtr'], d['ytr'])

        model.eval()
        with torch.no_grad():
            Xte_t = torch.tensor(d['Xte']).to(DEVICE)
            pred  = model(Xte_t).cpu().numpy()

        y_actual = price_scaler.inverse_transform(d['yte'].reshape(-1,1)).flatten()
        y_pred   = price_scaler.inverse_transform(pred.reshape(-1,1)).flatten()

        mae  = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mape = np.mean(np.abs((y_actual - y_pred)/y_actual)) * 100

        all_results[mname][h]   = dict(mae=mae, rmse=rmse, mape=mape,
                                        y_test=y_actual, y_pred=y_pred)
        all_histories[mname][h] = dict(train=tr_loss, val=val_loss)

        print(f"MAE=${mae:,.0f} | RMSE=${rmse:,.0f} | MAPE={mape:.2f}%")

print("\n✅ All models trained!")


# ─────────────────────────────────────────────
# STEP 9: RESULTS TABLE
# ─────────────────────────────────────────────
print("\n" + "="*68)
print(f"{'Model':<14} {'Horizon':<9} {'MAE':>11} {'RMSE':>11} {'MAPE':>9}")
print("="*68)
for m in MODEL_CLASSES:
    for h in HORIZONS:
        r = all_results[m][h]
        print(f"{m:<14} {str(h)+'D':<9} ${r['mae']:>10,.0f} ${r['rmse']:>10,.0f} {r['mape']:>8.2f}%")
    print("-"*68)


# ─────────────────────────────────────────────
# STEP 10: VISUALIZATIONS
# ─────────────────────────────────────────────
COLORS = ['#FF6B35','#4ECDC4','#45B7D1','#96CEB4']

# --- Actual vs Predicted ---
fig, axes = plt.subplots(4, 3, figsize=(18, 14))
fig.suptitle('Actual vs Predicted Bitcoin Price', fontsize=14, fontweight='bold')

for i, mname in enumerate(MODEL_CLASSES):
    for j, h in enumerate(HORIZONS):
        ax = axes[i][j]
        r  = all_results[mname][h]
        ax.plot(r['y_test'], color='black', lw=1.5, label='Actual', alpha=0.8)
        ax.plot(r['y_pred'], color=COLORS[i], lw=1.5, ls='--', label='Predicted')
        ax.set_title(f"{mname} | {h}D | MAPE:{r['mape']:.1f}%", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p: f'${v:,.0f}'))

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: actual_vs_predicted.png")

# --- Loss Curves ---
fig, axes = plt.subplots(4, 3, figsize=(18, 14))
fig.suptitle('Training & Validation Loss', fontsize=14, fontweight='bold')

for i, mname in enumerate(MODEL_CLASSES):
    for j, h in enumerate(HORIZONS):
        ax = axes[i][j]
        h_data = all_histories[mname][h]
        ax.plot(h_data['train'], color=COLORS[i], lw=1.5, label='Train')
        ax.plot(h_data['val'],   color='gray',     lw=1.5, ls='--', label='Val')
        ax.set_title(f"{mname} | {h}D", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: loss_curves.png")

# --- MAE Bar Chart ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('MAE Comparison', fontsize=13, fontweight='bold')
mnames = list(MODEL_CLASSES.keys())
x = np.arange(len(mnames))

for j, h in enumerate(HORIZONS):
    ax   = axes[j]
    maes = [all_results[m][h]['mae'] for m in mnames]
    bars = ax.bar(x, maes, color=COLORS, edgecolor='black', alpha=0.85)
    ax.set_title(f'{h}-Day Horizon', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(mnames, rotation=15, ha='right')
    ax.set_ylabel('MAE (USD)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,p: f'${v:,.0f}'))
    ax.grid(alpha=0.3, axis='y')
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('mae_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: mae_comparison.png")

# --- Error Distribution ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Error Distribution — 1D Horizon', fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, mname in enumerate(MODEL_CLASSES):
    r = all_results[mname][1]
    errors = r['y_test'] - r['y_pred']
    axes[i].hist(errors, bins=40, color=COLORS[i], edgecolor='black', alpha=0.8)
    axes[i].axvline(0, color='red', ls='--', lw=1.5)
    axes[i].set_title(f'{mname} | Mean Error: ${errors.mean():,.0f}')
    axes[i].set_xlabel('Error (USD)'); axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('error_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: error_distribution.png")


# ─────────────────────────────────────────────
# STEP 11: BEST MODEL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("🏆 BEST MODEL PER HORIZON (Lowest MAPE)")
print("="*55)
for h in HORIZONS:
    best = min(MODEL_CLASSES.keys(), key=lambda m: all_results[m][h]['mape'])
    r    = all_results[best][h]
    print(f"  {h}D → {best:12s} MAE=${r['mae']:,.0f} | MAPE={r['mape']:.2f}%")

print("\n✅ Project Complete!")
print("📁 Files saved: eda_plots.png | actual_vs_predicted.png")
print("               loss_curves.png | mae_comparison.png | error_distribution.png")
print("\n💡 Upload .py + all .png files to GitHub for submission!")
