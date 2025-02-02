import numpy as np
import pandas as pd
import math
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import pickle

plt.style.use('default')

# Veri yükleme ve işleme
veri1 = pd.read_csv('usd.csv')
veri1 = veri1[['Date','TRY']]
veri1['Date'] = pd.to_datetime(veri1['Date'], format='%Y-%m-%d')
veri1 = veri1.sort_values(by=['Date'], ascending=True)

# Verileri normalize etme
values = veri1['TRY'].values.reshape(-1, 1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(values)

# Eğitim ve test veri setlerine ayırma
TRAIN_SIZE = 0.70
train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Veri setini oluşturma fonksiyonu
def create_dataset(dataset, window_size=1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return np.array(data_X), np.array(data_Y)

window_size = 1
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)

# DNN için verileri yeniden şekillendirme
train_X_dnn = train_X.reshape(train_X.shape[0], -1)
test_X_dnn = test_X.reshape(test_X.shape[0], -1)

# LSTM için verileri yeniden şekillendirme
train_X_lstm = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X_lstm = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# LSTM modeli
def fit_lstm_model(train_X_lstm, train_Y, window_size=1):
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, input_shape=(1, window_size)))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss="mean_squared_error", optimizer="adam")
    lstm_model.fit(train_X_lstm, train_Y, epochs=5, batch_size=1, verbose=1)
    return lstm_model

# DNN modeli
def fit_dnn_model(train_X_dnn, train_Y, window_size=1):
    dnn_model = Sequential()
    dnn_model.add(Dense(64, input_dim=window_size, activation='relu'))
    dnn_model.add(Dense(32, activation='relu'))
    dnn_model.add(Dense(1))
    dnn_model.compile(loss='mean_squared_error', optimizer='adam')
    dnn_model.fit(train_X_dnn, train_Y, epochs=5, batch_size=1, verbose=1)
    return dnn_model

lstm_model = fit_lstm_model(train_X_lstm, train_Y, window_size)
dnn_model = fit_dnn_model(train_X_dnn, train_Y, window_size)

# Tahmin ve performans ölçümü yapan fonksiyon
def predict_and_score(model, X, Y, scaler):
    pred = scaler.inverse_transform(model.predict(X))
    orig_data = scaler.inverse_transform([Y])
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return score, pred

rmse_train_lstm, train_predict_lstm = predict_and_score(lstm_model, train_X_lstm, train_Y, scaler)
rmse_test_lstm, test_predict_lstm = predict_and_score(lstm_model, test_X_lstm, test_Y, scaler)

rmse_train_dnn, train_predict_dnn = predict_and_score(dnn_model, train_X_dnn, train_Y, scaler)
rmse_test_dnn, test_predict_dnn = predict_and_score(dnn_model, test_X_dnn, test_Y, scaler)

print(f"Eğitim verisi RMSE (LSTM): {rmse_train_lstm:.2f}")
print(f"Test verisi RMSE (LSTM): {rmse_test_lstm:.2f}")

print(f"Eğitim verisi RMSE (DNN): {rmse_train_dnn:.2f}")
print(f"Test verisi RMSE (DNN): {rmse_test_dnn:.2f}")

# Eğitim ve test tahminlerini plot yapma
train_predict_plot_lstm = np.empty_like(dataset)
train_predict_plot_lstm[:, :] = np.nan
train_predict_plot_lstm[window_size:len(train_predict_lstm) + window_size, :] = train_predict_lstm

test_predict_plot_lstm = np.empty_like(dataset)
test_predict_plot_lstm[:, :] = np.nan
test_predict_plot_lstm[len(train_predict_lstm) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict_lstm

# LSTM tahminleri
fig, ax1 = plt.subplots(figsize=(16, 8))
ax1.set_title("LSTM Modeli")
ax1.plot(veri1['Date'], scaler.inverse_transform(dataset), label="Gerçek Değerler")
ax1.plot(veri1['Date'][window_size:len(train_predict_lstm) + window_size], train_predict_lstm, label="Eğitim Tahminleri")
ax1.plot(veri1['Date'][len(train_predict_lstm) + (window_size * 2) + 1:len(dataset) - 1], test_predict_lstm, label="Test Tahminleri")
ax1.set_ylabel('Değerler')
ax1.set_xlabel('Tarih')
plt.legend()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.xticks(fontsize=8)  # Font boyutunu küçültme
plt.show()

# DNN tahminleri
fig, ax2 = plt.subplots(figsize=(16, 8))
ax2.set_title("DNN Modeli")
ax2.plot(veri1['Date'], scaler.inverse_transform(dataset), label="Gerçek Değerler")
ax2.plot(veri1['Date'][window_size:len(train_predict_dnn) + window_size], train_predict_dnn, label="Eğitim Tahminleri")
ax2.plot(veri1['Date'][len(train_predict_dnn) + (window_size * 2) + 1:len(dataset) - 1], test_predict_dnn, label="Test Tahminleri")
ax2.set_ylabel('Değerler')
ax2.set_xlabel('Tarih')
plt.legend()
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.xticks(fontsize=8)
plt.show()

# Karşılaştırma Tablosu
comparison_data = {
    "Model": ["LSTM", "DNN"],
    "RMSE (Eğitim)": [rmse_train_lstm, rmse_train_dnn],
    "RMSE (Test)": [rmse_test_lstm, rmse_test_dnn]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nModellerin Performans Karşılaştırması:")
print(comparison_df)

# Modellerin başarı oranları
baseline_rmse = np.std(scaler.inverse_transform(dataset))  # Standart sapmaya göre hesaplama
success_rate_train_lstm = 100 * (1 - (rmse_train_lstm / baseline_rmse))
success_rate_test_lstm = 100 * (1 - (rmse_test_lstm / baseline_rmse))

success_rate_train_dnn = 100 * (1 - (rmse_train_dnn / baseline_rmse))
success_rate_test_dnn = 100 * (1 - (rmse_test_dnn / baseline_rmse))

success_rates = {
    "Model": ["LSTM", "DNN"],
    "Başarı Oranı (Eğitim)": [success_rate_train_lstm, success_rate_train_dnn],
    "Başarı Oranı (Test)": [success_rate_test_lstm, success_rate_test_dnn]
}

success_df = pd.DataFrame(success_rates)
print("\nModellerin Başarı Oranları:")
print(success_df)

# Modelleri kaydet
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
lstm_model.save('lstm_model.keras')
dnn_model.save('dnn_model.keras')
print("\nlstm_model.keras, dnn_model.keras ve scaler.pkl model dosyaları kaydedildi.")