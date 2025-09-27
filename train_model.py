import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt

# === 1. CSV 읽기 ===
df = pd.read_csv('original.csv')

# === 2. 입력(X), 출력(y) 분리 ===
X = df[['AoA', 
        'CST Coeff 1','CST Coeff 2','CST Coeff 3','CST Coeff 4',
        'CST Coeff 5','CST Coeff 6','CST Coeff 7','CST Coeff 8']]

y_cl = df['Cl'].values
y_cd = df['Cd'].values

# === 3. 데이터 분할 ===
X_train, X_test, y_cl_train, y_cl_test, y_cd_train, y_cd_test = train_test_split(
    X, y_cl, y_cd, test_size=0.2, random_state=42)

# === 4. 표준화 ===
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
joblib.dump(scaler_X, 'scaler_X.pkl')

# Cl 스케일링
scaler_cl = StandardScaler()
y_cl_train_scaled = scaler_cl.fit_transform(y_cl_train.reshape(-1,1))
y_cl_test_scaled = scaler_cl.transform(y_cl_test.reshape(-1,1))
joblib.dump(scaler_cl, 'scaler_cl.pkl')

# Cd는 XGBoost에서 스케일링 없이 사용 가능 (tree-based는 스케일에 덜 민감)
# 필요하면 scaler_cd 사용 가능

# === 5. Cl MLP 모델 ===
inputs = Input(shape=(9,))
x = Dense(256, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1)(x)
model_cl = Model(inputs=inputs, outputs=outputs)
model_cl.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history_cl = model_cl.fit(X_train_scaled, y_cl_train_scaled,
                          validation_split=0.2,
                          epochs=300,
                          batch_size=256,
                          callbacks=[early_stop],
                          verbose=1)

# === 6. Cd XGBoost 모델 ===
model_cd = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model_cd.fit(X_train_scaled, y_cd_train)

# === 7. 예측 ===
# Cl 예측
y_cl_pred_scaled = model_cl.predict(X_test_scaled)
y_cl_pred = scaler_cl.inverse_transform(y_cl_pred_scaled)

# Cd 예측
y_cd_pred = model_cd.predict(X_test_scaled)

# === 8. 평가 ===
mse_cl = mean_squared_error(y_cl_test, y_cl_pred)
mae_cl = mean_absolute_error(y_cl_test, y_cl_pred)
r2_cl = r2_score(y_cl_test, y_cl_pred)

mse_cd = mean_squared_error(y_cd_test, y_cd_pred)
mae_cd = mean_absolute_error(y_cd_test, y_cd_pred)
r2_cd = r2_score(y_cd_test, y_cd_pred)

print(f"Cl - MSE: {mse_cl:.5f}, MAE: {mae_cl:.5f}, R²: {r2_cl:.5f}")
print(f"Cd - MSE: {mse_cd:.5f}, MAE: {mae_cd:.5f}, R²: {r2_cd:.5f}")

# === 9. 그래프 자동 저장 ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('graphs', exist_ok=True)

# Cl 학습 곡선
plt.figure(figsize=(6,4))
plt.plot(history_cl.history['loss'], label='Train Loss')
plt.plot(history_cl.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs'); plt.ylabel('MSE')
plt.title('Cl Training & Validation Loss'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(f'graphs/Cl_loss_{timestamp}.png', dpi=300)
plt.close()

# Cl 실제 vs 예측
plt.figure(figsize=(5,5))
plt.scatter(y_cl_test, y_cl_pred, alpha=0.3)
plt.plot([y_cl_test.min(), y_cl_test.max()],
         [y_cl_test.min(), y_cl_test.max()], 'r--')
plt.xlabel('Actual Cl'); plt.ylabel('Predicted Cl')
plt.title('Cl: Actual vs Predicted'); plt.grid(True)
plt.tight_layout()
plt.savefig(f'graphs/Cl_actual_vs_pred_{timestamp}.png', dpi=300)
plt.close()

# Cd 실제 vs 예측
plt.figure(figsize=(5,5))
plt.scatter(y_cd_test, y_cd_pred, alpha=0.3)
plt.plot([y_cd_test.min(), y_cd_test.max()],
         [y_cd_test.min(), y_cd_test.max()], 'r--')
plt.xlabel('Actual Cd'); plt.ylabel('Predicted Cd')
plt.title('Cd: Actual vs Predicted'); plt.grid(True)
plt.tight_layout()
plt.savefig(f'graphs/Cd_actual_vs_pred_{timestamp}.png', dpi=300)
plt.close()

# === 10. 모델 저장 ===
model_cl.save('airfoil_cl_model.h5')
joblib.dump(model_cd, 'airfoil_cd_model.pkl')  # XGBoost 모델은 pickle 사용
print("✅ Cl & Cd 하이브리드 모델과 그래프가 저장되었습니다.")
