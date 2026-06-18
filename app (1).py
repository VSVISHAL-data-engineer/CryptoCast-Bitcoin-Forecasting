"""
╔══════════════════════════════════════════════════════════════╗
║   CryptoCast — Streamlit App                                 ║
║   Multi-Horizon Bitcoin Price Forecasting                    ║
║   Run with: streamlit run app.py                             ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="CryptoCast — Bitcoin Forecasting",
                   page_icon="₿", layout="wide")

DEVICE = torch.device('cpu')


# ─────────────────────────────────────────────
# MODEL DEFINITIONS (must match training script exactly)
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
    def forward(self, x):
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


MODEL_CLASSES = {'1D-CNN': CNNModel, 'RNN': RNNModel,
                 'LSTM': LSTMModel, 'Transformer': TransformerModel}
MODEL_FILE_PREFIX = {'1D-CNN': 'CNN', 'RNN': 'RNN', 'LSTM': 'LSTM', 'Transformer': 'Transformer'}


# ─────────────────────────────────────────────
# CACHED LOADERS (so the app doesn't reload everything on every click)
# ─────────────────────────────────────────────
@st.cache_resource
def load_config_and_scalers():
    with open('saved_models/config.pkl', 'rb') as f:
        config = pickle.load(f)
    with open('saved_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('saved_models/price_scaler.pkl', 'rb') as f:
        price_scaler = pickle.load(f)
    return config, scaler, price_scaler


@st.cache_data
def load_data():
    data = pd.read_csv('saved_models/cleaned_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data


@st.cache_data
def load_results():
    return pd.read_csv('saved_models/results_summary.csv')


@st.cache_resource
def load_model(model_name, horizon, seq_len, n_feat):
    fname = f"saved_models/{MODEL_FILE_PREFIX[model_name]}_{horizon}D.pt"
    model = MODEL_CLASSES[model_name](seq_len, n_feat)
    model.load_state_dict(torch.load(fname, map_location=DEVICE))
    model.eval()
    return model


# ─────────────────────────────────────────────
# CHECK IF MODELS EXIST
# ─────────────────────────────────────────────
if not os.path.exists('saved_models/config.pkl'):
    st.error("⚠️ Models not found! Please run `python train_and_save_models.py` first "
              "to train and save the models before launching this app.")
    st.stop()

config, scaler, price_scaler = load_config_and_scalers()
data = load_data()
results_df = load_results()

SEQ_LEN  = config['SEQ_LEN']
FEATURES = config['FEATURES']
HORIZONS = config['HORIZONS']
N_FEAT   = config['N_FEAT']


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("₿ CryptoCast — Multi-Horizon Bitcoin Price Forecasting")
st.caption("Deep Learning models (1D-CNN, RNN, LSTM, Transformer) trained to forecast "
           "Bitcoin prices 1, 3, and 7 days ahead.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Price", f"${data['Price'].iloc[-1]:,.2f}")
col2.metric("Latest Date", data['Date'].iloc[-1].strftime('%d-%b-%Y'))
col3.metric("Data Points", f"{len(data):,}")
col4.metric("Date Range", f"{data['Date'].iloc[0].year}–{data['Date'].iloc[-1].year}")

st.divider()


# ─────────────────────────────────────────────
# SIDEBAR — USER CONTROLS
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Forecast Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model Architecture",
    list(MODEL_CLASSES.keys()),
    help="Pick which deep learning architecture to use for the prediction."
)

horizon_choice = st.sidebar.radio(
    "Choose Forecast Horizon",
    HORIZONS,
    format_func=lambda h: f"{h}-Day Ahead",
    help="How many days into the future to forecast."
)

st.sidebar.divider()
st.sidebar.markdown("**Model Performance (on test set)**")
row = results_df[(results_df['model'] == model_choice) & (results_df['horizon'] == horizon_choice)].iloc[0]
st.sidebar.metric("MAE", f"${row['mae']:,.0f}")
st.sidebar.metric("RMSE", f"${row['rmse']:,.0f}")
st.sidebar.metric("MAPE", f"{row['mape']:.2f}%")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📊 Historical Data & EDA", "🏆 Model Comparison"])


# --- TAB 1: LIVE PREDICTION ---
with tab1:
    st.subheader(f"Forecast using {model_choice} — {horizon_choice}-Day Horizon")

    model = load_model(model_choice, horizon_choice, SEQ_LEN, N_FEAT)

    # Take the most recent SEQ_LEN days from the dataset as input
    recent_data = data[FEATURES].tail(SEQ_LEN).values
    scaled_input = scaler.transform(recent_data)
    input_tensor = torch.tensor(scaled_input, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()

    pred_price = price_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
    last_price = data['Price'].iloc[-1]
    change = pred_price - last_price
    change_pct = (change / last_price) * 100

    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Current Price", f"${last_price:,.2f}")
    pcol2.metric(f"Predicted Price ({horizon_choice}D ahead)",
                f"${pred_price:,.2f}", f"{change_pct:+.2f}%")
    pcol3.metric("Expected Change", f"${change:,.2f}")

    st.info(f"📅 Based on the last **{SEQ_LEN} days** of data (up to "
            f"{data['Date'].iloc[-1].strftime('%d-%b-%Y')}), the **{model_choice}** model "
            f"predicts Bitcoin will be priced at **${pred_price:,.2f}** in **{horizon_choice} day(s)**.")

    # Plot recent trend + predicted point
    fig, ax = plt.subplots(figsize=(12, 5))
    recent_plot = data.tail(90)
    ax.plot(recent_plot['Date'], recent_plot['Price'], label='Historical Price',
           color='steelblue', linewidth=1.8)

    future_date = data['Date'].iloc[-1] + pd.Timedelta(days=horizon_choice)
    ax.scatter([future_date], [pred_price], color='red', s=120, zorder=5,
              label=f'Predicted ({horizon_choice}D ahead)')
    ax.plot([data['Date'].iloc[-1], future_date], [last_price, pred_price],
           color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_title('Recent Price Trend + Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)

    st.caption("⚠️ This is a model-generated forecast for educational purposes only — "
              "not financial advice. Cryptocurrency prices are highly volatile.")


# --- TAB 2: HISTORICAL DATA & EDA ---
with tab2:
    st.subheader("Historical Bitcoin Data")

    date_range = st.slider(
        "Select date range to view",
        min_value=data['Date'].min().to_pydatetime(),
        max_value=data['Date'].max().to_pydatetime(),
        value=(data['Date'].iloc[-365].to_pydatetime(), data['Date'].max().to_pydatetime())
    )
    filtered = data[(data['Date'] >= date_range[0]) & (data['Date'] <= date_range[1])]

    ecol1, ecol2 = st.columns(2)

    with ecol1:
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(filtered['Date'], filtered['Price'], color='orange', linewidth=1.3)
        ax1.set_title('Closing Price')
        ax1.grid(alpha=0.3)
        fig1.autofmt_xdate()
        st.pyplot(fig1)

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.fill_between(filtered['Date'], filtered['Vol.'], alpha=0.6, color='steelblue')
        ax3.set_title('Trading Volume')
        ax3.grid(alpha=0.3)
        fig3.autofmt_xdate()
        st.pyplot(fig3)

    with ecol2:
        colors_bar = ['green' if x >= 0 else 'red' for x in filtered['Change %']]
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.bar(filtered['Date'], filtered['Change %'], color=colors_bar, alpha=0.7, width=1)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_title('Daily Change %')
        ax2.grid(alpha=0.3)
        fig2.autofmt_xdate()
        st.pyplot(fig2)

        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.hist(filtered['Price'], bins=40, color='orange', edgecolor='black', alpha=0.7)
        ax4.set_title('Price Distribution')
        ax4.grid(alpha=0.3)
        st.pyplot(fig4)

    st.subheader("Raw Data Table")
    st.dataframe(filtered.sort_values('Date', ascending=False), use_container_width=True)


# --- TAB 3: MODEL COMPARISON ---
with tab3:
    st.subheader("Model Performance Comparison")

    pivot_mae = results_df.pivot(index='model', columns='horizon', values='mae')
    pivot_mape = results_df.pivot(index='model', columns='horizon', values='mape')

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown("**MAE (USD) — lower is better**")
        st.dataframe(pivot_mae.style.format("${:,.0f}").background_gradient(cmap='RdYlGn_r', axis=None),
                    use_container_width=True)
    with mcol2:
        st.markdown("**MAPE (%) — lower is better**")
        st.dataframe(pivot_mape.style.format("{:.2f}%").background_gradient(cmap='RdYlGn_r', axis=None),
                    use_container_width=True)

    st.subheader("MAPE by Model & Horizon")
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(HORIZONS))
    width = 0.2
    colors_m = ['#FF6B35','#4ECDC4','#45B7D1','#96CEB4']

    for i, m in enumerate(MODEL_CLASSES.keys()):
        vals = [results_df[(results_df['model']==m) & (results_df['horizon']==h)]['mape'].values[0] for h in HORIZONS]
        ax5.bar(x + i*width, vals, width=width, label=m, color=colors_m[i])

    ax5.set_xticks(x + width*1.5)
    ax5.set_xticklabels([f'{h}D' for h in HORIZONS])
    ax5.set_ylabel('MAPE (%)')
    ax5.set_title('Forecast Error by Model & Horizon')
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')
    st.pyplot(fig5)

    best_rows = results_df.loc[results_df.groupby('horizon')['mape'].idxmin()]
    st.subheader("🏆 Best Model per Horizon")
    st.dataframe(best_rows[['horizon', 'model', 'mae', 'rmse', 'mape']]
                .rename(columns={'horizon':'Horizon (days)', 'model':'Best Model',
                                 'mae':'MAE ($)', 'rmse':'RMSE ($)', 'mape':'MAPE (%)'}),
                use_container_width=True, hide_index=True)
