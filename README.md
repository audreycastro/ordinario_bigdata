# Proyecto Final de Machine Learning: Regresión y Clasificación

Este repositorio contiene el desarrollo de un proyecto completo de Machine Learning, abarcando desde la ingestión y análisis de datos hasta el entrenamiento y la evaluación de modelos predictivos para tareas de regresión y clasificación.

 Audrey Castro

---

## Objetivo

El objetivo principal es desarrollar un flujo de trabajo completo, desde la ingestión de datos hasta la interpretación de modelos predictivos, documentando cada paso en un formato claro y reproducible. El proyecto cubre dos problemas fundamentales del aprendizaje automático:

1.  **Regresión:** Predecir los costos de seguros médicos personales.
2.  **Clasificación:** Predecir la presencia de enfermedades cardíacas en pacientes.

## Datasets Utilizados

Se utilizaron dos datasets públicos de Kaggle y el repositorio de la UCI:

1.  **Regresion - Medical Cost Personal Datasets:**
    *   **Descripción:** Contiene información demográfica y de salud de personas para predecir sus costos médicos.
    *   **Enlace:** [Kaggle - Medical Cost](https://www.kaggle.com/datasets/mirichoi0218/insurance)

2.  **Clasificacion - Heart Disease UCI:**
    *   **Descripción:** Base de datos clínica para la predicción de enfermedades cardíacas.
    *   **Enlace:** [Kaggle - Heart Disease](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

## Estructura del Repositorio

El proyecto está organizado en las siguientes carpetas:



data/                 # Contiene los datasets en formato .csv
 insurance.csv
 heart_processed.csv
     images/               # Almacena los gráficos generados (EDA, evaluación)
      correlation_matrix.png
      eda_regresion.png
      model_evaluation_regresion.png
      correlation_matrix_clasificacion.png
       eda_clasificacion.png
       confusion_matrix_logistic.png
       confusion_matrix_tree.png
       model_comparison.png
 notebooks/            # Scripts de Python con el código de cada análisis
 1_regresion_medical_cost.py
 2_clasificacion_heart_disease.py
 README.md             # Este archivo, con la documentación del proyecto


## Cómo Ejecutar el Proyecto


    

1.  **Instalar las dependencias:**
    Se recomienda crear un entorno virtual.
 
    pip install pandas numpy matplotlib seaborn scikit-learn
    

2.  **Ejecutar los scripts:**
    Navega a la carpeta `notebooks` y ejecuta los archivos de Python en orden.
    
    cd notebooks
    python 1_regresion_medical_cost.py
    python 2_clasificacion_heart_disease.py

    Los resultados, incluyendo métricas y gráficos, se imprimirán en la consola y se guardarán en la carpeta images.



## 5. Reporte de Resultados

### Resumen de Hallazgos

#### Análisis de Regresión (Costos Médicos)

El objetivo era predecir los costos de seguros médicos. Se utilizó un modelo de **Regresión Lineal**.

*   **Variables más influyentes:** El análisis de correlación y los coeficientes del modelo revelaron que ser **fumador (smoker_yes)** es, con diferencia, el factor que más incrementa el costo del seguro. Otras variables importantes son la **edad (age)** y el **índice de masa corporal (bmi)**.
*   **Rendimiento del modelo:** El modelo logró un **R² Score de 0.78** en el conjunto de prueba, lo que significa que explica aproximadamente el 78% de la variabilidad en los costos médicos. El **Error Cuadrático Medio (RMSE)** fue de **$5,796.28**, indicando el error promedio de las predicciones.

#### Análisis de Clasificación (Enfermedad Cardíaca)

El objetivo era predecir la presencia o ausencia de enfermedad cardíaca. Se entrenaron y compararon dos modelos: **Regresión Logística** y **Árbol de Decisión**.

*   **Variables más influyentes:** Las variables con mayor correlación con la enfermedad cardíaca fueron el **tipo de dolor de pecho (cp)**, la **depresión del segmento ST (oldpeak)**, la **frecuencia cardíaca máxima alcanzada (thalach)** y la presencia de **angina inducida por ejercicio (exang)**.
*   **Rendimiento del modelo:**
    *   La **Regresión Logística** (clasificador_castro) demostró ser el mejor modelo, con un **Accuracy del 83.3%**, una **Precisión del 84.6%** y un **Recall del 78.6%**.
    *   El **Árbol de Decisión** tuvo un rendimiento inferior en el conjunto de prueba, con un 70% de Accuracy, sugiriendo un posible sobreajuste al conjunto de entrenamiento.

### Comparación y Dificultades

*   **¿Qué modelo fue más difícil de ajustar?**
    El modelo de **clasificación de enfermedad cardíaca** presentó una mayor complejidad. El dataset original requirió una limpieza más cuidadosa, incluyendo la adición de encabezados, la gestión de valores faltantes representados como `?` y la transformación de la variable objetivo a un formato binario (0 o 1). Además, la selección entre Regresión Logística y Árbol de Decisión implicó una comparación de métricas (Accuracy, Precision, Recall) para determinar cuál ofrecía un mejor equilibrio, especialmente importante en un contexto médico donde los falsos negativos (no detectar a un enfermo) pueden ser críticos.

    En contraste, el modelo de **regresión de costos médicos** fue más directo. El dataset estaba limpio y las relaciones, aunque no perfectamente lineales, fueron bien capturadas por el modelo de regresión lineal, que sirvió como una excelente base de análisis.

