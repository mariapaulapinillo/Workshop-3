# 🌍 World Happiness Streaming Pipeline — Workshop 3 (UAO)

Proyecto académico desarrollado como parte del curso **ETL y Machine Learning**, Universidad Autónoma de Occidente.

## 🚀 Descripción
Pipeline completo para predecir el *Happiness Score* de los países del informe mundial de felicidad (2015–2019).  
Integra ETL, entrenamiento de modelos de regresión, y procesamiento **en tiempo real** con **Apache Kafka**.

---

## 🧩 Estructura del proyecto

| Etapa | Archivo / Notebook | Descripción |
|--------|--------------------|--------------|
| EDA | `EDA.ipynb` | Análisis exploratorio y selección de variables |
| ETL | `extract.py`, `transform.py`, `load.py` | Limpieza, unificación y carga de datos |
| Model Training | `train_model.py` | Entrenamiento de modelos (OLS y Random Forest) |
| Streaming | `producer.py`, `consumer.py`, `kafka_config.py` | Envío y consumo de datos en tiempo real |
| Evaluation | `evaluate_model.py` | Métricas de desempeño y validación final |
| Visualización | `performance_visuals.py` | Gráficos finales de evaluación |

---
Document: https://docs.google.com/document/d/1K7gS8SQY9E84ULTJIWow2Q12j3ytfjlvLENGVKFVsP0/edit?usp=sharing
