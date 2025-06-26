import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP
import os

# -------------------------------
# Función de preprocesamiento de datos
# -------------------------------
def preprocess_data(df):
    df = df.copy()
    
    # Si existe una columna de índice llamada 'Nº', la eliminamos
    if 'Nº' in df.columns:
        df = df.drop(columns=['Nº'])
    
    # Mapeo de variables categóricas a valores numéricos
    df['A1'] = df['A1'].map({'malo': 0, 'conocido': 1, 'bueno': 2})
    df['A2'] = df['A2'].map({'alto': 0, 'bajo': 1})
    df['A3'] = df['A3'].map({'ninguno': 0, 'adecuado': 1})
    
    # Seleccionamos las características de entrada
    X = df[['A1', 'A2', 'A3', 'A4']].values
    
    # Normalizamos las características para que estén en el mismo rango
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)
    
    # Codificamos la columna 'Clase' en formato one-hot para entrenamiento
    y = df['Clase']
    y_encoded = pd.get_dummies(y).values
    
    return X_norm, y_encoded, X_min, X_max

# -------------------------------
# Interfaz de entrenamiento
# -------------------------------
st.title("🏦 Sistema de Clasificación de Clientes con Red Neuronal")
st.markdown("**Entrene el modelo con un dataset etiquetado (con columna 'Clase')**")

# Subida de archivo de entrenamiento
uploaded_file = st.file_uploader("Subir dataset de entrenamiento (CSV o Excel)", type=["csv", "xlsx"], key="train")
if uploaded_file:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        # Cargamos el archivo según su extensión
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == '.xlsx':
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de archivo no soportado. Use CSV o Excel.")
            st.stop()
            
        st.success("Datos cargados correctamente!")
        st.dataframe(df.head())
        
        # Verificamos que estén todas las columnas necesarias
        required_columns = ['A1', 'A2', 'A3', 'A4', 'Clase']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"Faltan columnas requeridas: {', '.join(missing)}")
            st.stop()
        
        # Preprocesamos los datos
        X, y, X_min, X_max = preprocess_data(df)
        
        # Selección de hiperparámetros en la barra lateral
        st.sidebar.header("⚙️ Hiperparámetros")
        learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.001, 0.5, 0.01)
        epochs = st.sidebar.slider("Épocas", 100, 5000, 1000)
        hidden_neurons = st.sidebar.slider("Neuronas ocultas", 2, 20, 5)
        
        # Entrenamiento del modelo al presionar el botón
        if st.button("🔧 Entrenar Modelo"):
            # Creamos la red neuronal con la arquitectura definida
            model = MLP(input_size=4, hidden_size=hidden_neurons, output_size=3)
            
            with st.spinner("Entrenando red neuronal..."):
                loss_history = model.train(X, y, learning_rate, epochs)
            
            # Guardamos el modelo y los parámetros de normalización en session_state
            st.session_state.model = model
            st.session_state.X_min = X_min
            st.session_state.X_max = X_max
            
            # Mostramos la evolución de la pérdida durante el entrenamiento
            fig, ax = plt.subplots()
            ax.plot(loss_history)
            ax.set_title("Evolución de la Pérdida durante el Entrenamiento")
            ax.set_xlabel("Época")
            ax.set_ylabel("Pérdida")
            st.pyplot(fig)
            
            # Mostramos la precisión final sobre los datos de entrenamiento
            accuracy = model.accuracy(X, y)
            st.success(f"✅ Precisión en datos de entrenamiento: {accuracy*100:.2f}%")
            st.write("**Nota:** 1=Malo, 2=Promedio, 3=Bueno")
    
    except Exception as e:
        st.error(f"Error procesando archivo: {str(e)}")

# -------------------------------
# Predicción masiva de nuevos clientes
# -------------------------------
st.header("📂 Predicción Masiva de Clientes")
uploaded_pred = st.file_uploader("Subir dataset para predicción (sin columna 'Clase')", type=["csv", "xlsx"], key="predict")

if uploaded_pred and 'model' in st.session_state:
    file_extension = os.path.splitext(uploaded_pred.name)[1].lower()
    try:
        # Cargamos el archivo de predicción
        if file_extension == '.csv':
            df_pred = pd.read_csv(uploaded_pred)
        elif file_extension == '.xlsx':
            df_pred = pd.read_excel(uploaded_pred)
        else:
            st.error("Formato de archivo no soportado. Use CSV o Excel.")
            st.stop()

        # Verificamos que estén todas las columnas necesarias (sin 'Clase')
        required_columns = ['A1', 'A2', 'A3', 'A4']
        if not all(col in df_pred.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df_pred.columns]
            st.error(f"Faltan columnas requeridas: {', '.join(missing)}")
            st.stop()

        # Mapeamos las variables categóricas igual que en el entrenamiento
        df_pred['A1'] = df_pred['A1'].map({'malo': 0, 'conocido': 1, 'bueno': 2})
        df_pred['A2'] = df_pred['A2'].map({'alto': 0, 'bajo': 1})
        df_pred['A3'] = df_pred['A3'].map({'ninguno': 0, 'adecuado': 1})

        # Normalizamos usando los parámetros del entrenamiento
        X_pred = df_pred[['A1', 'A2', 'A3', 'A4']].values
        X_pred_norm = (X_pred - st.session_state.X_min) / (st.session_state.X_max - st.session_state.X_min + 1e-8)

        # Predecimos la clase para cada cliente
        pred_clases = st.session_state.model.predict(X_pred_norm)
        df_pred['Clase_Predicha'] = pred_clases

        st.success("Predicción realizada. Aquí están los resultados:")
        st.dataframe(df_pred)

        # ----------- Análisis de resultados -----------

        # 1. Gráfico: cantidad de clientes por clase predicha
        st.subheader("Distribución de Clientes por Clase Predicha")
        clase_counts = df_pred['Clase_Predicha'].value_counts().sort_index()
        clase_labels = {1: "MALO", 2: "PROMEDIO", 3: "BUENO"}
        fig1, ax1 = plt.subplots()
        ax1.bar([clase_labels.get(i, i) for i in clase_counts.index], clase_counts.values, color=['red', 'orange', 'green'])
        ax1.set_xlabel("Clase Predicha")
        ax1.set_ylabel("Cantidad de Clientes")
        ax1.set_title("Cantidad de Clientes por Clase Predicha")
        st.pyplot(fig1)

        # 2. Gráfico: características más frecuentes en los nuevos clientes
        st.subheader("Características Más Frecuentes en Nuevos Clientes")
        # Diccionarios para revertir el mapeo numérico a texto
        inv_map_A1 = {0: 'malo', 1: 'conocido', 2: 'bueno'}
        inv_map_A2 = {0: 'alto', 1: 'bajo'}
        inv_map_A3 = {0: 'ninguno', 1: 'adecuado'}

        # Si los datos están mapeados, revertimos para mostrar etiquetas
        df_show = df_pred.copy()
        if df_show['A1'].dtype != object:
            df_show['A1'] = df_show['A1'].map(inv_map_A1)
        if df_show['A2'].dtype != object:
            df_show['A2'] = df_show['A2'].map(inv_map_A2)
        if df_show['A3'].dtype != object:
            df_show['A3'] = df_show['A3'].map(inv_map_A3)

        # Contamos la frecuencia de cada valor en cada característica
        freq_A1 = df_show['A1'].value_counts()
        freq_A2 = df_show['A2'].value_counts()
        freq_A3 = df_show['A3'].value_counts()
        freq_A4 = df_show['A4'].value_counts()

        # Mostramos los gráficos de barras para cada característica
        fig2, axs = plt.subplots(1, 4, figsize=(16, 4))
        axs[0].bar(freq_A1.index, freq_A1.values, color='blue')
        axs[0].set_title("A1 (Historia)")
        axs[1].bar(freq_A2.index, freq_A2.values, color='purple')
        axs[1].set_title("A2 (Pasivo)")
        axs[2].bar(freq_A3.index, freq_A3.values, color='brown')
        axs[2].set_title("A3 (Garantía)")
        axs[3].bar(freq_A4.index.astype(str), freq_A4.values, color='gray')
        axs[3].set_title("A4 (Ganancia)")
        for ax in axs:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        fig2.suptitle("Frecuencia de Características en Nuevos Clientes")
        st.pyplot(fig2)

        # Permite descargar los resultados con la clase predicha
        csv = df_pred.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar resultados como CSV", csv, "predicciones.csv", "text/csv")

    except Exception as e:
        st.error(f"Error procesando archivo: {str(e)}")
elif uploaded_pred:
    st.warning("Primero debe entrenar el modelo con un dataset etiquetado.")

# -------------------------------
# Clasificación de un solo cliente (simulación)
# -------------------------------
st.header("🔮 Clasificar Nuevo Cliente")
st.subheader("Ingrese las características del cliente:")

if 'model' in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        a1 = st.selectbox("Historia de créditos", options=['malo', 'conocido', 'bueno'])
        a2 = st.selectbox("Pasivo", options=['alto', 'bajo'])
    with col2:
        a3 = st.selectbox("Garantía", options=['ninguno', 'adecuado'])
        a4 = st.slider("Ganancia", 1, 3, 2)
    
    if st.button("🏷️ Predecir Clasificación"):
        # Preparamos los datos del cliente para el modelo
        input_data = pd.DataFrame([[a1, a2, a3, a4]], columns=['A1', 'A2', 'A3', 'A4'])
        
        # Mapeamos igual que en el entrenamiento
        input_data['A1'] = input_data['A1'].map({'malo':0, 'conocido':1, 'bueno':2})
        input_data['A2'] = input_data['A2'].map({'alto':0, 'bajo':1})
        input_data['A3'] = input_data['A3'].map({'ninguno':0, 'adecuado':1})
        
        # Normalizamos usando los parámetros del entrenamiento
        input_norm = (input_data.values - st.session_state.X_min) / (st.session_state.X_max - st.session_state.X_min + 1e-8)
        
        # Predecimos la clase (1, 2, 3)
        prediction = st.session_state.model.predict(input_norm)[0]
        
        # Mostramos el resultado de la predicción
        st.subheader("Resultado de la Clasificación")
        if prediction == 1:
            st.error(f"🔴 Cliente MALO (Puntuación: 1)")
            st.markdown("**Riesgo alto:** Historial crediticio negativo, pasivos altos y garantías insuficientes")
        elif prediction == 2:
            st.warning(f"🟡 Cliente PROMEDIO (Puntuación: 2)")
            st.markdown("**Riesgo moderado:** Características mixtas que requieren evaluación adicional")
        else:
            st.success(f"🟢 Cliente BUENO (Puntuación: 3)")
            st.markdown("**Riesgo bajo:** Historial crediticio positivo, pasivos bajos y garantías adecuadas")
        
        # Barra de progreso visual
        st.progress(prediction/3)
        
        # Detalles técnicos para revisión
        with st.expander("Ver detalles técnicos"):
            st.subheader("Características procesadas")
            st.json({
                "A1 (Historia créditos)": f"{a1} → {input_data['A1'].values[0]}",
                "A2 (Pasivo)": f"{a2} → {input_data['A2'].values[0]}",
                "A3 (Garantía)": f"{a3} → {input_data['A3'].values[0]}",
                "A4 (Ganancia)": a4
            })
            st.subheader("Valores normalizados")
            st.write(input_norm)
else:
    st.warning("⚠️ Primero cargue un dataset de entrenamiento y entrene un modelo")

# -------------------------------
# Información sobre el modelo en la barra lateral
# -------------------------------
st.sidebar.header("ℹ️ Acerca del Modelo")
st.sidebar.markdown("""
Este sistema utiliza una red neuronal multicapa implementada completamente desde cero (sin librerías de ML) para clasificar clientes en 3 categorías:

1. 🔴 **Cliente MALO** (1): Alto riesgo crediticio
2. 🟡 **Cliente PROMEDIO** (2): Riesgo moderado
3. 🟢 **Cliente BUENO** (3): Bajo riesgo crediticio

El modelo se entrena con datos históricos que incluyen 4 características:
- Historia crediticia (A1)
- Nivel de pasivos (A2)
- Garantías (A3)
- Nivel de ganancia (A4)
""")