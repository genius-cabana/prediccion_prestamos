# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

print("🚀 Iniciando análisis del dataset...")

# 📁 1. Cargar los datos
df = pd.read_csv('data/dataBasePrestDigital.csv', sep=';')

# Renombrar columnas para facilitar trabajo
df.columns = [
    'mes', 'cliente', 'estado_cliente', 'rango_edad', 'genero',
    'rango_sueldo', 'procedencia', 'canal_digital', 'transacciones_mes',
    'promedio_transacciones_3m', 'recurrencia_campania', 'frecuencia_campania',
    'tiene_tarjeta', 'prom_cons_banco_3m', 'prom_saldo_banco_3m',
    'prom_saldo_tc_3m', 'prom_saldo_prest_3m', 'sow_tc', 'sow_prestamo', 'compra_prestamo_digital'
]

# Convertir mes a datetime
df['mes'] = pd.to_datetime(df['mes'].astype(str), format='%Y%m')

# 🧮 2. Función para parsear rango de sueldo
def parse_rango_sueldo(rango):
    if isinstance(rango, str):
        rango = rango.strip().lower()
        if '<' in rango and '-' in rango:
            try:
                lim_inf, lim_sup = map(float, rango.replace('<', '').replace(']', '').split('-'))
                return (lim_inf + lim_sup) / 2
            except:
                return np.nan
        elif '>' in rango:
            try:
                lim_inf = float(rango.replace('>', '').strip())
                return lim_inf + 500
            except:
                return np.nan
        elif '<' in rango and not '-' in rango:
            try:
                lim_sup = float(rango.replace('<', '').replace(']', ''))
                return lim_sup / 2
            except:
                return np.nan
        else:
            return np.nan
    return np.nan

# Crear columna numérica de sueldo
df['sueldo_promedio'] = df['rango_sueldo'].apply(parse_rango_sueldo)

# Convertir variable objetivo a numérico
df['prom_saldo_prest_3m'] = pd.to_numeric(df['prom_saldo_prest_3m'], errors='coerce')

# Eliminar filas con NaN en la variable objetivo
df = df.dropna(subset=['prom_saldo_prest_3m'])

# 📊 3. Regresión Lineal Simple
# Usamos sueldo como variable independiente
X_simple = df[['sueldo_promedio']].copy()
y_simple = df['prom_saldo_prest_3m'].copy()

# Limpiar juntas para asegurar consistencia
combined = pd.concat([X_simple, y_simple], axis=1)
combined_clean = combined.dropna()

X_simple_clean = combined_clean[['sueldo_promedio']]
y_simple_clean = combined_clean['prom_saldo_prest_3m']

# Dividir datos
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_simple_clean, y_simple_clean, test_size=0.2, random_state=42
)

# Entrenar modelo
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)
y_pred_lr = lr_model.predict(X_test_lr)

# Graficar resultados
plt.figure(figsize=(10,6))
plt.scatter(X_test_lr, y_test_lr, color='blue', label='Real')
plt.plot(X_test_lr, y_pred_lr, color='red', label='Predicción LR')
plt.title('Regresión Lineal: Sueldo vs Saldo Promedio Préstamo')
plt.xlabel('Sueldo Promedio')
plt.ylabel('Saldo Promedio Préstamo')
plt.legend()
plt.grid(True)
os.makedirs('images', exist_ok=True)
plt.savefig('images/regresion_lineal.png')
plt.show()

# 🤖 4. Red Neuronal Artificial (ANN)
# Variables predictoras
X = df[['sueldo_promedio', 'rango_edad', 'genero', 'procedencia', 'canal_digital',
        'transacciones_mes', 'promedio_transacciones_3m', 'recurrencia_campania',
        'frecuencia_campania', 'tiene_tarjeta', 'prom_cons_banco_3m',
        'prom_saldo_banco_3m', 'sow_tc', 'sow_prestamo']]

# Variable objetivo
y = df['prom_saldo_prest_3m']

# Preprocesamiento
categorical_cols = ['rango_edad', 'genero', 'procedencia', 'canal_digital', 'tiene_tarjeta']
numerical_cols = X.columns.drop(categorical_cols).tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)

# Asegurar que y sea numérico
y = pd.to_numeric(y, errors='coerce')

# Eliminar filas con NaN
mask = ~np.isnan(y) & ~np.isnan(X_processed).any(axis=1)
X_clean = X_processed[mask]
y_clean = y[mask]

# Dividir datos
X_train_ann, X_test_ann, y_train_ann, y_test_ann = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

# Construir red neuronal
model_ann = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_ann.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model_ann.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar modelo
history = model_ann.fit(X_train_ann, y_train_ann, epochs=50, batch_size=32,
                        validation_data=(X_test_ann, y_test_ann), verbose=0)

# Graficar pérdida
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Evolución del error durante entrenamiento')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('images/loss_vs_epochs_ann.png')
plt.show()

# Comparar predicciones vs reales
y_pred_ann = model_ann.predict(X_test_ann).flatten()
plt.scatter(y_test_ann, y_pred_ann)
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Red Neuronal: Predicción vs Real')
plt.grid(True)
plt.savefig('images/prediccion_vs_real_ann.png')
plt.show()

# 🧩 5. Lógica de programación básica
# Ejemplo con bucle, condición, lista y diccionario
resultados = []

for i in range(len(y_test_ann)):
    diferencia = abs(y_test_ann.iloc[i] - y_pred_ann[i])
    categoria = "Alta" if diferencia > 1000 else "Baja"
    resultados.append({
        "indice": i,
        "real": y_test_ann.iloc[i],
        "predicho": y_pred_ann[i],
        "diferencia": diferencia,
        "categoria": categoria
    })

# Mostrar primeros 5 resultados
print("\n📊 Primeros 5 resultados:")
for res in resultados[:5]:
    print(res)

# 📦 6. Guarda métricas finales
mse = mean_squared_error(y_test_ann, y_pred_ann)
mae = mean_absolute_error(y_test_ann, y_pred_ann)
print(f"\n📈 Métricas finales:")
print(f"MSE: {mse:.2f}, MAE: {mae:.2f}")