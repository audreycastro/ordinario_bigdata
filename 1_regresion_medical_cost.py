#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proyecto de Machine Learning - Regresión
Dataset: Medical Cost Personal
Alumna: Audrey Castro
Objetivo: Predecir costos médicos usando regresión lineal
"""

# ============================================================================
# 1. CARGA Y SEGMENTACIÓN DE INFORMACIÓN (ETL)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")

print("="*70)
print("PROYECTO DE REGRESIÓN - MEDICAL COST PERSONAL")
print("Alumna: Audrey Castro")
print("="*70)

# Cargar dataset con nomenclatura personalizada
df_audrey = pd.read_csv('../data/insurance.csv')

print("\n1. CARGA Y SEGMENTACIÓN DE INFORMACIÓN (ETL)")
print("-"*70)
print("\n✓ Dataset cargado exitosamente como 'df_audrey'")
print(f"\n📊 Dimensiones del dataset: {df_audrey.shape[0]} filas x {df_audrey.shape[1]} columnas")

# Mostrar las primeras 5 filas (Requisito del proyecto)
print("\n🔍 Primeras 5 filas del dataset:")
print(df_audrey.head())

# Información general del dataset
print("\n📋 Información del dataset:")
print(df_audrey.info())

# Análisis de valores nulos
print("\n🔎 Análisis de valores nulos:")
valores_nulos = df_audrey.isnull().sum()
print(valores_nulos)

if valores_nulos.sum() == 0:
    print("\n✅ No se encontraron valores nulos en el dataset")
else:
    print(f"\n⚠️ Se encontraron {valores_nulos.sum()} valores nulos")

# Estadísticas descriptivas
print("\n📈 Estadísticas descriptivas:")
print(df_audrey.describe())

# ============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================

print("\n" + "="*70)
print("2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("-"*70)

# Crear figura para visualizaciones
fig = plt.figure(figsize=(16, 12))

# 2.1 Distribución de la variable objetivo (charges)
plt.subplot(3, 3, 1)
plt.hist(df_audrey['charges'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Costo Médico ($)', fontsize=10)
plt.ylabel('Frecuencia', fontsize=10)
plt.title('Distribución de Costos Médicos', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

# 2.2 Distribución de edad
plt.subplot(3, 3, 2)
plt.hist(df_audrey['age'], bins=20, edgecolor='black', alpha=0.7, color='coral')
plt.xlabel('Edad', fontsize=10)
plt.ylabel('Frecuencia', fontsize=10)
plt.title('Distribución de Edad', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

# 2.3 Distribución de BMI
plt.subplot(3, 3, 3)
plt.hist(df_audrey['bmi'], bins=25, edgecolor='black', alpha=0.7, color='lightgreen')
plt.xlabel('BMI (Índice de Masa Corporal)', fontsize=10)
plt.ylabel('Frecuencia', fontsize=10)
plt.title('Distribución de BMI', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

# 2.4 Relación Edad vs Costo
plt.subplot(3, 3, 4)
plt.scatter(df_audrey['age'], df_audrey['charges'], alpha=0.5, s=20)
plt.xlabel('Edad', fontsize=10)
plt.ylabel('Costo Médico ($)', fontsize=10)
plt.title('Edad vs Costo Médico', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

# 2.5 Relación BMI vs Costo
plt.subplot(3, 3, 5)
plt.scatter(df_audrey['bmi'], df_audrey['charges'], alpha=0.5, s=20, color='green')
plt.xlabel('BMI', fontsize=10)
plt.ylabel('Costo Médico ($)', fontsize=10)
plt.title('BMI vs Costo Médico', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

# 2.6 Costo por Fumador
plt.subplot(3, 3, 6)
df_audrey.boxplot(column='charges', by='smoker', ax=plt.gca())
plt.xlabel('Fumador', fontsize=10)
plt.ylabel('Costo Médico ($)', fontsize=10)
plt.title('Costo Médico por Fumador', fontsize=11, fontweight='bold')
plt.suptitle('')
plt.grid(True, alpha=0.3)

# 2.7 Costo por Sexo
plt.subplot(3, 3, 7)
df_audrey.boxplot(column='charges', by='sex', ax=plt.gca())
plt.xlabel('Sexo', fontsize=10)
plt.ylabel('Costo Médico ($)', fontsize=10)
plt.title('Costo Médico por Sexo', fontsize=11, fontweight='bold')
plt.suptitle('')
plt.grid(True, alpha=0.3)

# 2.8 Costo por Región
plt.subplot(3, 3, 8)
df_audrey.boxplot(column='charges', by='region', ax=plt.gca())
plt.xlabel('Región', fontsize=10)
plt.ylabel('Costo Médico ($)', fontsize=10)
plt.title('Costo Médico por Región', fontsize=11, fontweight='bold')
plt.suptitle('')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 2.9 Número de hijos
plt.subplot(3, 3, 9)
df_audrey['children'].value_counts().sort_index().plot(kind='bar', edgecolor='black', alpha=0.7)
plt.xlabel('Número de Hijos', fontsize=10)
plt.ylabel('Frecuencia', fontsize=10)
plt.title('Distribución de Número de Hijos', fontsize=11, fontweight='bold')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/eda_regresion.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráficos exploratorios guardados en '../images/eda_regresion.png'")
plt.close()

# Matriz de correlación (Requisito del proyecto)
print("\n📊 Generando matriz de correlación...")

# Convertir variables categóricas a numéricas para correlación
df_audrey_numeric = df_audrey.copy()
df_audrey_numeric['sex'] = df_audrey_numeric['sex'].map({'male': 1, 'female': 0})
df_audrey_numeric['smoker'] = df_audrey_numeric['smoker'].map({'yes': 1, 'no': 0})
df_audrey_numeric['region'] = df_audrey_numeric['region'].astype('category').cat.codes

# Calcular correlación
correlation_matrix = df_audrey_numeric.corr()

# Visualizar matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación - Variables vs Costos Médicos', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../images/correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Matriz de correlación guardada en '../images/correlation_matrix.png'")
plt.close()

print("\n🔍 Correlación con la variable objetivo (charges):")
print(correlation_matrix['charges'].sort_values(ascending=False))

# ============================================================================
# 3. MODELADO DE REGRESIÓN
# ============================================================================

print("\n" + "="*70)
print("3. MODELADO DE REGRESIÓN")
print("-"*70)

# Preprocesamiento: Convertir variables categóricas
print("\n🔧 Preprocesamiento de datos...")

df_model = df_audrey.copy()

# One-Hot Encoding para variables categóricas
df_model = pd.get_dummies(df_model, columns=['sex', 'smoker', 'region'], drop_first=True)

print(f"\n✓ Variables después de One-Hot Encoding: {df_model.shape[1]} columnas")
print(f"Columnas: {list(df_model.columns)}")

# Separar características (X) y variable objetivo (y)
X_acm = df_model.drop('charges', axis=1)
y_acm = df_model['charges']

# División en conjunto de entrenamiento y prueba (80-20)
# Nomenclatura personalizada: X_train_acm, X_test_acm, y_train_acm, y_test_acm
X_train_acm, X_test_acm, y_train_acm, y_test_acm = train_test_split(
    X_acm, y_acm, test_size=0.2, random_state=42
)

print(f"\n📊 División de datos:")
print(f"   - Entrenamiento: {X_train_acm.shape[0]} muestras")
print(f"   - Prueba: {X_test_acm.shape[0]} muestras")

# Entrenar modelo de Regresión Lineal
# Nomenclatura personalizada: modelo_castro
print("\n🤖 Entrenando modelo de Regresión Lineal...")

modelo_castro = LinearRegression()
modelo_castro.fit(X_train_acm, y_train_acm)

print("✓ Modelo entrenado exitosamente: 'modelo_castro'")

# Realizar predicciones
y_pred_train_acm = modelo_castro.predict(X_train_acm)
y_pred_test_acm = modelo_castro.predict(X_test_acm)

# ============================================================================
# 4. EVALUACIÓN DEL MODELO
# ============================================================================

print("\n" + "="*70)
print("4. EVALUACIÓN DEL MODELO")
print("-"*70)

# Calcular métricas
r2_train = r2_score(y_train_acm, y_pred_train_acm)
r2_test = r2_score(y_test_acm, y_pred_test_acm)
rmse_train = np.sqrt(mean_squared_error(y_train_acm, y_pred_train_acm))
rmse_test = np.sqrt(mean_squared_error(y_test_acm, y_pred_test_acm))

print("\n📈 MÉTRICAS DE EVALUACIÓN:")
print(f"\n   R² Score (Entrenamiento): {r2_train:.4f}")
print(f"   R² Score (Prueba):        {r2_test:.4f}")
print(f"\n   RMSE (Entrenamiento):     ${rmse_train:,.2f}")
print(f"   RMSE (Prueba):            ${rmse_test:,.2f}")

# Interpretación
print("\n" + "="*70)
print("5. INTERPRETACIÓN DE RESULTADOS")
print("-"*70)

# Coeficientes del modelo
coeficientes = pd.DataFrame({
    'Variable': X_acm.columns,
    'Coeficiente': modelo_castro.coef_
})
coeficientes = coeficientes.sort_values('Coeficiente', ascending=False, key=abs)

print("\n📊 COEFICIENTES DEL MODELO (ordenados por importancia):")
print(coeficientes.to_string(index=False))

print(f"\n📍 Intercepto del modelo: ${modelo_castro.intercept_:,.2f}")

print("\n" + "="*70)
print("CONCLUSIONES:")
print("-"*70)
print("""
1. VARIABLES MÁS INFLUYENTES:
   - Ser fumador (smoker_yes) es el factor que más incrementa los costos médicos
   - La edad también tiene un impacto significativo positivo
   - El BMI muestra una relación positiva con los costos

2. RENDIMIENTO DEL MODELO:
   - El R² indica que el modelo explica aproximadamente el {:.1f}% de la variabilidad
   - El RMSE de ${:,.2f} representa el error promedio de predicción

3. RECOMENDACIONES:
   - El modelo puede mejorarse con técnicas de regularización
   - Considerar interacciones entre variables (ej: edad * fumador)
   - Explorar modelos no lineales para capturar relaciones complejas
""".format(r2_test * 100, rmse_test))

# Visualización de predicciones
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Valores reales vs predichos
axes[0].scatter(y_test_acm, y_pred_test_acm, alpha=0.5, s=30)
axes[0].plot([y_test_acm.min(), y_test_acm.max()], 
             [y_test_acm.min(), y_test_acm.max()], 
             'r--', lw=2, label='Predicción perfecta')
axes[0].set_xlabel('Valores Reales ($)', fontsize=11)
axes[0].set_ylabel('Valores Predichos ($)', fontsize=11)
axes[0].set_title('Valores Reales vs Predichos\n(Conjunto de Prueba)', 
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Residuos
residuos = y_test_acm - y_pred_test_acm
axes[1].scatter(y_pred_test_acm, residuos, alpha=0.5, s=30, color='green')
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Valores Predichos ($)', fontsize=11)
axes[1].set_ylabel('Residuos ($)', fontsize=11)
axes[1].set_title('Gráfico de Residuos\n(Conjunto de Prueba)', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/model_evaluation_regresion.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráficos de evaluación guardados en '../images/model_evaluation_regresion.png'")
plt.close()

print("\n" + "="*70)
print("PROYECTO DE REGRESIÓN COMPLETADO")
print("="*70)
