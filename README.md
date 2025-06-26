# Clasificación de Clientes con Red Neuronal desde Cero

Este proyecto implementa un sistema de clasificación de clientes en tres categorías de riesgo (MALO, PROMEDIO, BUENO) utilizando una red neuronal multicapa (MLP) **programada completamente desde cero con NumPy**, sin usar librerías de machine learning externas.

## Objetivo

Clasificar clientes según su historial crediticio, pasivos, garantía y ganancia, utilizando una red neuronal entrenada a partir de datos históricos.

## Estructura del Proyecto

```
├── app.py              # Interfaz principal en Streamlit
├── mlp.py              # Implementación de la red neuronal MLP desde cero
├── requirements.txt    # (opcional) Requisitos del entorno
```


## Uso de la aplicación

### Entrenamiento

1. Sube un archivo CSV o Excel con las columnas: `A1`, `A2`, `A3`, `A4`, `Clase`.
2. Ajusta los hiperparámetros: tasa de aprendizaje, épocas y neuronas ocultas.
3. Presiona "Entrenar Modelo".
4. Observa la evolución de la pérdida y la precisión.

### Predicción Masiva

1. Sube un archivo sin columna "Clase".
2. El sistema clasificará cada cliente automáticamente.
3. Puedes descargar los resultados en formato CSV.
4. Se mostrarán gráficos de distribución de clases y características más frecuentes.

### Clasificación Individual

1. Ingresa manualmente los valores de A1 a A4.
2. Obtendrás la clase predicha y detalles del análisis.

## Variables utilizadas
De entrada
- **A1**: Historia de créditos (`malo`, `conocido`, `bueno`)
- **A2**: Pasivo (`alto`, `bajo`)
- **A3**: Garantía (`ninguno`, `adecuado`)
- **A4**: Ganancia (número entre 1 y 3)

De salida
- **Clase**: Resultado (`1=Malo`, `2=Promedio`, `3=Bueno`)


Requerimientos
- Python 
- Streamlit
- NumPy
- Pandas
- Matplotlib
