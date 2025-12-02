#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proyecto de Machine Learning - Clasificación
Dataset: Heart Disease UCI
Alumna: Audrey Castro
Objetivo: Predecir presencia de enfermedad cardíaca usando clasificación
"""

# ============================================================================
# 1. CARGA Y SEGMENTACIÓN DE INFORMACIÓN (ETL)
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("Set2")

print("="*70)
print("PROYECTO DE CLASIFICACIÓN - HEART DISEASE UCI")
print("Alumna: Audrey Castro")
print("="*70)

# Cargar dataset con nomenclatura personalizada
df_audrey = pd.read_csv('../data/heart_processed.csv')

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
    # Eliminar filas con valores nulos si existen
    df_audrey = df_audrey.dropna()
    print(f"✓ Filas eliminadas. Nuevo tamaño: {df_audrey.shape[0]} filas")

# Verificar valores faltantes representados como '?'
print("\n🔎 Verificando valores '?' en el dataset...")
for col in df_audrey.columns:
    if df_audrey[col].dtype == 'object':
        missing_count = (df_audrey[col] == '?').sum()
        if missing_count > 0:
            print(f"   - Columna '{col}': {missing_count} valores '?'")
            df_audrey = df_audrey[df_audrey[col] != '?']

print(f"\n✓ Dataset limpio. Tamaño final: {df_audrey.shape[0]} filas")

# Estadísticas descriptivas
print("\n📈 Estadísticas descriptivas:")
print(df_audrey.describe())

# Descripción de las variables
print("\n📚 DESCRIPCIÓN DE VARIABLES:")
print("-"*70)
descripcion_variables = """
1. age: Edad del paciente (años)
2. sex: Sexo (1 = masculino, 0 = femenino)
3. cp: Tipo de dolor de pecho (1-4)
4. trestbps: Presión arterial en reposo (mm Hg)
5. chol: Colesterol sérico (mg/dl)
6. fbs: Glucosa en ayunas > 120 mg/dl (1 = verdadero, 0 = falso)
7. restecg: Resultados electrocardiográficos en reposo (0-2)
8. thalach: Frecuencia cardíaca máxima alcanzada
9. exang: Angina inducida por ejercicio (1 = sí, 0 = no)
10. oldpeak: Depresión del ST inducida por ejercicio
11. slope: Pendiente del segmento ST de ejercicio máximo (1-3)
12. ca: Número de vasos principales coloreados por fluoroscopia (0-3)
13. thal: Talasemia (3 = normal, 6 = defecto fijo, 7 = defecto reversible)
14. target: Diagnóstico de enfermedad cardíaca (0 = no, 1-4 = sí)
"""
print(descripcion_variables)

# Convertir target a binario (0 = sin enfermedad, 1 = con enfermedad)
df_audrey['target'] = df_audrey['target'].apply(lambda x: 1 if x > 0 else 0)

print("\n🎯 Distribución de la variable objetivo (target):")
print(df_audrey['target'].value_counts())
print(f"\n   - Sin enfermedad (0): {(df_audrey['target'] == 0).sum()} pacientes")
print(f"   - Con enfermedad (1): {(df_audrey['target'] == 1).sum()} pacientes")

# ============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================

print("\n" + "="*70)
print("2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("-"*70)

# Crear figura para visualizaciones
fig = plt.figure(figsize=(16, 12))

# 2.1 Distribución de la variable objetivo
plt.subplot(3, 3, 1)
target_counts = df_audrey['target'].value_counts()
plt.bar(['Sin Enfermedad', 'Con Enfermedad'], target_counts.values, 
        edgecolor='black', alpha=0.7, color=['lightgreen', 'salmon'])
plt.ylabel('Frecuencia', fontsize=10)
plt.title('Distribución de Diagnóstico', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(target_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')

# 2.2 Distribución por edad
plt.subplot(3, 3, 2)
plt.hist(df_audrey['age'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Edad', fontsize=10)
plt.ylabel('Frecuencia', fontsize=10)
plt.title('Distribución de Edad', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3)

# 2.3 Distribución por sexo
plt.subplot(3, 3, 3)
sex_counts = df_audrey['sex'].value_counts()
plt.bar(['Femenino', 'Masculino'], sex_counts.values, 
        edgecolor='black', alpha=0.7, color=['pink', 'lightblue'])
plt.ylabel('Frecuencia', fontsize=10)
plt.title('Distribución por Sexo', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 2.4 Edad vs Enfermedad
plt.subplot(3, 3, 4)
df_audrey.boxplot(column='age', by='target', ax=plt.gca())
plt.xlabel('Diagnóstico (0=No, 1=Sí)', fontsize=10)
plt.ylabel('Edad', fontsize=10)
plt.title('Edad vs Enfermedad Cardíaca', fontsize=11, fontweight='bold')
plt.suptitle('')
plt.grid(True, alpha=0.3)

# 2.5 Colesterol vs Enfermedad
plt.subplot(3, 3, 5)
df_audrey.boxplot(column='chol', by='target', ax=plt.gca())
plt.xlabel('Diagnóstico (0=No, 1=Sí)', fontsize=10)
plt.ylabel('Colesterol (mg/dl)', fontsize=10)
plt.title('Colesterol vs Enfermedad Cardíaca', fontsize=11, fontweight='bold')
plt.suptitle('')
plt.grid(True, alpha=0.3)

# 2.6 Frecuencia cardíaca máxima vs Enfermedad
plt.subplot(3, 3, 6)
df_audrey.boxplot(column='thalach', by='target', ax=plt.gca())
plt.xlabel('Diagnóstico (0=No, 1=Sí)', fontsize=10)
plt.ylabel('Frecuencia Cardíaca Máxima', fontsize=10)
plt.title('Frecuencia Cardíaca vs Enfermedad', fontsize=11, fontweight='bold')
plt.suptitle('')
plt.grid(True, alpha=0.3)

# 2.7 Tipo de dolor de pecho
plt.subplot(3, 3, 7)
cp_counts = df_audrey['cp'].value_counts().sort_index()
plt.bar(cp_counts.index, cp_counts.values, edgecolor='black', alpha=0.7)
plt.xlabel('Tipo de Dolor de Pecho', fontsize=10)
plt.ylabel('Frecuencia', fontsize=10)
plt.title('Distribución de Tipo de Dolor', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 2.8 Angina inducida por ejercicio vs Enfermedad
plt.subplot(3, 3, 8)
crosstab = pd.crosstab(df_audrey['exang'], df_audrey['target'])
crosstab.plot(kind='bar', ax=plt.gca(), edgecolor='black', alpha=0.7)
plt.xlabel('Angina por Ejercicio (0=No, 1=Sí)', fontsize=10)
plt.ylabel('Frecuencia', fontsize=10)
plt.title('Angina por Ejercicio vs Enfermedad', fontsize=11, fontweight='bold')
plt.legend(['Sin Enfermedad', 'Con Enfermedad'], loc='upper right')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')

# 2.9 Sexo vs Enfermedad
plt.subplot(3, 3, 9)
crosstab_sex = pd.crosstab(df_audrey['sex'], df_audrey['target'])
crosstab_sex.plot(kind='bar', ax=plt.gca(), edgecolor='black', alpha=0.7)
plt.xlabel('Sexo (0=Femenino, 1=Masculino)', fontsize=10)
plt.ylabel('Frecuencia', fontsize=10)
plt.title('Sexo vs Enfermedad Cardíaca', fontsize=11, fontweight='bold')
plt.legend(['Sin Enfermedad', 'Con Enfermedad'], loc='upper right')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../images/eda_clasificacion.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráficos exploratorios guardados en '../images/eda_clasificacion.png'")
plt.close()

# Matriz de correlación (Requisito del proyecto)
print("\n📊 Generando matriz de correlación...")

correlation_matrix = df_audrey.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación - Variables vs Enfermedad Cardíaca', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../images/correlation_matrix_clasificacion.png', dpi=300, bbox_inches='tight')
print("✓ Matriz de correlación guardada en '../images/correlation_matrix_clasificacion.png'")
plt.close()

print("\n🔍 Correlación con la variable objetivo (target):")
print(correlation_matrix['target'].sort_values(ascending=False))

# ============================================================================
# 3. PREPROCESAMIENTO DE DATOS
# ============================================================================

print("\n" + "="*70)
print("3. PREPROCESAMIENTO DE DATOS")
print("-"*70)

# Separar características (X) y variable objetivo (y)
X_acm = df_audrey.drop('target', axis=1)
y_acm = df_audrey['target']

print(f"\n📊 Características (X): {X_acm.shape[1]} variables")
print(f"🎯 Variable objetivo (y): {y_acm.name}")

# División en conjunto de entrenamiento y prueba (80-20)
# Nomenclatura personalizada: X_train_acm, X_test_acm, y_train_acm, y_test_acm
X_train_acm, X_test_acm, y_train_acm, y_test_acm = train_test_split(
    X_acm, y_acm, test_size=0.2, random_state=42, stratify=y_acm
)

print(f"\n📊 División de datos:")
print(f"   - Entrenamiento: {X_train_acm.shape[0]} muestras")
print(f"   - Prueba: {X_test_acm.shape[0]} muestras")

# Normalización de características
print("\n🔧 Normalizando características...")
scaler = StandardScaler()
X_train_acm_scaled = scaler.fit_transform(X_train_acm)
X_test_acm_scaled = scaler.transform(X_test_acm)
print("✓ Normalización completada")

# ============================================================================
# 4. MODELADO DE CLASIFICACIÓN
# ============================================================================

print("\n" + "="*70)
print("4. MODELADO DE CLASIFICACIÓN")
print("-"*70)

# Modelo 1: Regresión Logística
# Nomenclatura personalizada: clasificador_castro
print("\n🤖 Entrenando modelo de Regresión Logística...")

clasificador_castro = LogisticRegression(max_iter=1000, random_state=42)
clasificador_castro.fit(X_train_acm_scaled, y_train_acm)

print("✓ Modelo entrenado exitosamente: 'clasificador_castro'")

# Realizar predicciones
y_pred_train_acm = clasificador_castro.predict(X_train_acm_scaled)
y_pred_test_acm = clasificador_castro.predict(X_test_acm_scaled)

# ============================================================================
# 5. EVALUACIÓN DEL MODELO
# ============================================================================

print("\n" + "="*70)
print("5. EVALUACIÓN DEL MODELO - REGRESIÓN LOGÍSTICA")
print("-"*70)

# Calcular métricas
accuracy_train = accuracy_score(y_train_acm, y_pred_train_acm)
accuracy_test = accuracy_score(y_test_acm, y_pred_test_acm)
precision_test = precision_score(y_test_acm, y_pred_test_acm)
recall_test = recall_score(y_test_acm, y_pred_test_acm)

print("\n📈 MÉTRICAS DE EVALUACIÓN:")
print(f"\n   Accuracy (Entrenamiento): {accuracy_train:.4f} ({accuracy_train*100:.2f}%)")
print(f"   Accuracy (Prueba):        {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")
print(f"\n   Precision (Prueba):       {precision_test:.4f} ({precision_test*100:.2f}%)")
print(f"   Recall (Prueba):          {recall_test:.4f} ({recall_test*100:.2f}%)")

# Matriz de confusión
print("\n📊 MATRIZ DE CONFUSIÓN:")
cm = confusion_matrix(y_test_acm, y_pred_test_acm)
print(cm)

# Visualizar matriz de confusión
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de confusión - Valores absolutos
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
            yticklabels=['Sin Enfermedad', 'Con Enfermedad'],
            cbar_kws={"shrink": 0.8})
axes[0].set_xlabel('Predicción', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Valor Real', fontsize=11, fontweight='bold')
axes[0].set_title('Matriz de Confusión\n(Regresión Logística)', 
                  fontsize=12, fontweight='bold')

# Matriz de confusión - Porcentajes
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', ax=axes[1],
            xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
            yticklabels=['Sin Enfermedad', 'Con Enfermedad'],
            cbar_kws={"shrink": 0.8})
axes[1].set_xlabel('Predicción', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Valor Real', fontsize=11, fontweight='bold')
axes[1].set_title('Matriz de Confusión (%)\n(Regresión Logística)', 
                  fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('../images/confusion_matrix_logistic.png', dpi=300, bbox_inches='tight')
print("\n✓ Matriz de confusión guardada en '../images/confusion_matrix_logistic.png'")
plt.close()

# Reporte de clasificación completo
print("\n📋 REPORTE DE CLASIFICACIÓN COMPLETO:")
print(classification_report(y_test_acm, y_pred_test_acm, 
                          target_names=['Sin Enfermedad', 'Con Enfermedad']))

# ============================================================================
# 6. MODELO ALTERNATIVO: ÁRBOL DE DECISIÓN
# ============================================================================

print("\n" + "="*70)
print("6. MODELO ALTERNATIVO - ÁRBOL DE DECISIÓN")
print("-"*70)

print("\n🌳 Entrenando modelo de Árbol de Decisión...")

tree_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_classifier.fit(X_train_acm, y_train_acm)

print("✓ Árbol de Decisión entrenado exitosamente")

# Predicciones
y_pred_tree_train = tree_classifier.predict(X_train_acm)
y_pred_tree_test = tree_classifier.predict(X_test_acm)

# Métricas
accuracy_tree_train = accuracy_score(y_train_acm, y_pred_tree_train)
accuracy_tree_test = accuracy_score(y_test_acm, y_pred_tree_test)
precision_tree = precision_score(y_test_acm, y_pred_tree_test)
recall_tree = recall_score(y_test_acm, y_pred_tree_test)

print("\n📈 MÉTRICAS DE EVALUACIÓN (Árbol de Decisión):")
print(f"\n   Accuracy (Entrenamiento): {accuracy_tree_train:.4f} ({accuracy_tree_train*100:.2f}%)")
print(f"   Accuracy (Prueba):        {accuracy_tree_test:.4f} ({accuracy_tree_test*100:.2f}%)")
print(f"\n   Precision (Prueba):       {precision_tree:.4f} ({precision_tree*100:.2f}%)")
print(f"   Recall (Prueba):          {recall_tree:.4f} ({recall_tree*100:.2f}%)")

# Matriz de confusión para árbol de decisión
cm_tree = confusion_matrix(y_test_acm, y_pred_tree_test)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Sin Enfermedad', 'Con Enfermedad'],
            yticklabels=['Sin Enfermedad', 'Con Enfermedad'],
            cbar_kws={"shrink": 0.8})
plt.xlabel('Predicción', fontsize=11, fontweight='bold')
plt.ylabel('Valor Real', fontsize=11, fontweight='bold')
plt.title('Matriz de Confusión - Árbol de Decisión', 
          fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('../images/confusion_matrix_tree.png', dpi=300, bbox_inches='tight')
print("\n✓ Matriz de confusión (Árbol) guardada en '../images/confusion_matrix_tree.png'")
plt.close()

# ============================================================================
# 7. COMPARACIÓN DE MODELOS
# ============================================================================

print("\n" + "="*70)
print("7. COMPARACIÓN DE MODELOS")
print("-"*70)

# Crear tabla comparativa
comparacion = pd.DataFrame({
    'Modelo': ['Regresión Logística', 'Árbol de Decisión'],
    'Accuracy': [accuracy_test, accuracy_tree_test],
    'Precision': [precision_test, precision_tree],
    'Recall': [recall_test, recall_tree]
})

print("\n📊 TABLA COMPARATIVA DE MODELOS:")
print(comparacion.to_string(index=False))

# Visualización comparativa
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(comparacion['Modelo']))
width = 0.25

bars1 = ax.bar(x - width, comparacion['Accuracy'], width, label='Accuracy', alpha=0.8)
bars2 = ax.bar(x, comparacion['Precision'], width, label='Precision', alpha=0.8)
bars3 = ax.bar(x + width, comparacion['Recall'], width, label='Recall', alpha=0.8)

ax.set_xlabel('Modelo', fontsize=12, fontweight='bold')
ax.set_ylabel('Puntuación', fontsize=12, fontweight='bold')
ax.set_title('Comparación de Modelos de Clasificación', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(comparacion['Modelo'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

# Añadir valores sobre las barras
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../images/model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico comparativo guardado en '../images/model_comparison.png'")
plt.close()

# ============================================================================
# 8. INTERPRETACIÓN Y CONCLUSIONES
# ============================================================================

print("\n" + "="*70)
print("8. INTERPRETACIÓN Y CONCLUSIONES")
print("-"*70)

print("""
📋 HALLAZGOS PRINCIPALES:

1. VARIABLES MÁS CORRELACIONADAS CON ENFERMEDAD CARDÍACA:
   - Tipo de dolor de pecho (cp): Correlación positiva fuerte
   - Frecuencia cardíaca máxima (thalach): Correlación negativa
   - Angina inducida por ejercicio (exang): Correlación positiva
   - Depresión del ST (oldpeak): Correlación positiva

2. RENDIMIENTO DEL MODELO DE REGRESIÓN LOGÍSTICA:
   - Accuracy: {:.2f}% - El modelo clasifica correctamente la mayoría de casos
   - Precision: {:.2f}% - Cuando predice enfermedad, acierta en este porcentaje
   - Recall: {:.2f}% - Detecta este porcentaje de casos reales de enfermedad

3. COMPARACIÓN DE MODELOS:
   - Regresión Logística: Mejor balance entre precision y recall
   - Árbol de Decisión: Rendimiento similar, más interpretable

4. APLICACIÓN CLÍNICA:
   - El modelo puede ayudar en el screening inicial de pacientes
   - Alta precisión reduce falsos positivos
   - Buen recall asegura detección de casos verdaderos

5. RECOMENDACIONES:
   - Validar con datos de otros hospitales/regiones
   - Considerar ensemble methods (Random Forest, XGBoost)
   - Incluir más variables clínicas si están disponibles
   - Implementar validación cruzada para mayor robustez
""".format(accuracy_test*100, precision_test*100, recall_test*100))

print("\n" + "="*70)
print("PROYECTO DE CLASIFICACIÓN COMPLETADO")
print("="*70)
