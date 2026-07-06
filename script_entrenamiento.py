import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, auc
)

def evaluar_modelo():
    print("1. Cargando datos procesados desde la Fase 1...")
    ruta_data = os.path.join("data", "processed", "02_loan_data_clean.csv")
    
    if not os.path.exists(ruta_data):
        print(f"Error: No se encuentra el archivo {ruta_data}. Ejecuta primero scriptLimpieza.py")
        return
        
    df = pd.read_csv(ruta_data)

    # 2. Preprocesamiento: Convertir variables categóricas de texto a numéricas (One-Hot Encoding)
    # Esto procesará columnas como person_gender, person_education, loan_intent
    cat_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 3. Separar características (X) y variable objetivo (y)
    X = df_encoded.drop(columns=['loan_status'])
    y = df_encoded['loan_status']

    #la proporción 70-30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Datos de entrenamiento: {X_train.shape[0]} filas. Datos de prueba: {X_test.shape[0]} filas.")

    # 5. Inicializar y entrenar el modelo Random Forest
    print("3. Entrenando el modelo Random Forest (Algoritmo de Ensamble)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 6. Realizar predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 7. Calcular las métricas exigidas por la rúbrica de la EV3
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    gini = 2 * roc_auc - 1  # Relación matemática Gini-AUC

    # Imprimir resultados en consola (para tus logs del informe)
    print("\n================ METRICAS OBTENIDAS ================")
    print(f"Accuracy (Exactitud):       {acc:.4f}")
    print(f"Precision (Precisión):     {prec:.4f}")
    print(f"Recall (Sensibilidad):      {rec:.4f}")
    print(f"F1-Score:                   {f1:.4f}")
    print(f"AUC-ROC:                    {roc_auc:.4f}")
    print(f"Coeficiente de Gini:        {gini:.4f}")
    print("====================================================\n")

    # 8. Guardar gráficos automáticamente para el informe técnico
    print("4. Generando gráficos de rendimiento...")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Matriz de Confusión
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False)
    ax[0].set_title('Matriz de Confusión - Validación')
    ax[0].set_xlabel('Predicción del Modelo')
    ax[0].set_ylabel('Clase Real (0=Pagó, 1=Insolvente)')

    # Curva ROC
    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_title('Curva ROC')
    ax[1].set_xlabel('Tasa de Falsos Positivos (FPR)')
    ax[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)')
    ax[1].legend(loc="lower right")

    # Crear carpeta para guardar imágenes si no existe
    os.makedirs("docs", exist_ok=True)
    ruta_grafico = os.path.join("docs", "metricas_rendimiento.png")
    plt.tight_layout()
    plt.savefig(ruta_grafico)
    print(f"Gráficos guardados exitosamente en: {ruta_grafico}")

    # 9. EXPORTACIÓN DE DATOS PARA POWER BI
    print("5. Exportando resultados y métricas para Power BI...")
    
    # Creamos un DataFrame para el set de prueba con los resultados reales y las predicciones
    df_powerbi = X_test.copy()
    df_powerbi['loan_status_real'] = y_test.values
    df_powerbi['loan_status_pred'] = y_pred
    df_powerbi['probabilidad_impago'] = y_proba
    
    # Añadimos columnas de métricas globales fijas para que Power BI las lea directamente como KPI Cards
    df_powerbi['metric_accuracy'] = acc
    df_powerbi['metric_gini'] = gini
    df_powerbi['metric_auc_roc'] = roc_auc
    df_powerbi['metric_recall'] = rec
    
    # Guardar en la carpeta processed
    ruta_pbi_csv = os.path.join("data", "processed", "03_model_predictions_and_metrics.csv")
    df_powerbi.to_csv(ruta_pbi_csv, index=False)
    print(f"Archivo para Power BI generado exitosamente en: {ruta_pbi_csv}")

if __name__ == "__main__":
    evaluar_modelo()