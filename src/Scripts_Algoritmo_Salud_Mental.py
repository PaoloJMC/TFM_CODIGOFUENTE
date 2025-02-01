import pandas as pd
import numpy as np
from scipy import stats
import random
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
#import squarify
%matplotlib inline
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score



warnings.filterwarnings("ignore")

import pandas as pd

# Cargar los datos
df=pd.read_csv('train_depression.csv') 

# Revisión general
print(df.head())


print(df.info())
print(f'Número de filas y columnas: {df.shape}')

print("Checking if there are any null values:\n", df.isnull().sum())
print("\nChecking if there are any duplicate rows: ", df.duplicated().sum())

df.info()

# Extracting important column types
target_column = 'Depression'
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(exclude=['object']).columns.drop(target_column)

print("\n🔍 Column Types:")
print("Target Column:", target_column)
print("Categorical Columns:", categorical_columns.tolist())
print("Numerical Columns:", numerical_columns.tolist())

df.describe().round(2).style.format(precision=2).background_gradient(
    cmap="Blues"
)


#Encontramos  Muchas Clases en Dietary Habits , deberia salir 3 
df.describe(include="object").round(2).style.format(precision=2).background_gradient(
    cmap="Blues"
)


import matplotlib.pyplot as plt

# Distribuciones de las variables
df.hist(bins=30, figsize=(10, 10))
plt.show()

import math
from matplotlib import pyplot

# Número de columnas en el dataset
num_columns = len(df.columns)

# Calcular el número de filas y columnas necesarias para el layout
nrows = math.ceil(num_columns / 2)  # Por ejemplo, dividir en 2 columnas
ncols = 2  # Fijar 2 columnas

# Crear los boxplots
df.plot(kind='box', subplots=True, layout=(nrows, ncols), sharex=False, sharey=False, figsize=(12, nrows * 3))
pyplot.tight_layout()
pyplot.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Filtrar solo las columnas numéricas del DataFrame
df_numeric = df.select_dtypes(include=['float64', 'int64'])

#Correlacion entre nuestras variables features numericas.
correlation = df_numeric.corr()

# Set the figure size for the heatmap plot
plt.figure(figsize=(10, 6))

# Create a heatmap using seaborn (sns) to visualize the correlation matrix
# The 'annot' parameter displays the correlation values in each cell
sns.heatmap(correlation, annot=True)

# Set the title of the heatmap plot
plt.title("Correlation Matrix")

# Display the heatmap plot
plt.show()

# Gráficos de dispersión entre pares de variables
sns.pairplot(df)
plt.show()


# Seleccionar solo columnas categóricas
categorical_columns = df.select_dtypes(include=['object', 'category'])

for col in categorical_columns.columns:
    print(f"Valores únicos y frecuencias en la columna '{col}':")
    print(df[col].value_counts())
    print("\n")
    
# Gráfico de barras
#df['columna_categórica'].value_counts().plot(kind='bar')
#plt.show()

def imputar_valores_nulos(df):
    # Calcular la media y el modo para cada columna
    mean_values = df.select_dtypes(include=['float64', 'int64']).mean()
    mode_values = df.select_dtypes(include=['object']).mode().iloc[0]
    
    # Imputar valores nulos en columnas numéricas
    for col, value in mean_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    
    # Imputar valores nulos en columnas categóricas
    for col, value in mode_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    
    return df

df = imputar_valores_nulos(df)
df.head()

# Ejemplo de identificación e imputación de outliers
def identificar_imputar_outliers(df, metodo='IQR', factor=1.5):
    if metodo == 'IQR':
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - factor * IQR
            upper_limit = Q3 + factor * IQR
            
            # Imputar outliers con la mediana de la columna
            median_value = df[col].median()
            df[col] = np.where((df[col] < lower_limit) | (df[col] > upper_limit), median_value, df[col])
    
    return df


df = identificar_imputar_outliers(df)
df.head()

# Ejemplo de mapeo de categorías de duración del sueño
def mapear_duracion_sueno(df):
    sleep_mapping = {
        'More than 8 hours': 'High Sleep', '7-8 hours': 'High Sleep', '6-7 hours': 'Moderate Sleep', 
        '6-8 hours': 'Moderate Sleep', '5-6 hours': 'Moderate Sleep', '4-6 hours': 'Low Sleep', 
        '4-5 hours': 'Low Sleep', '3-4 hours': 'Low Sleep', '2-3 hours': 'Very Low Sleep', 
        '1-2 hours': 'Very Low Sleep', 'Less than 5 hours': 'Very Low Sleep', '10-11 hours': 'High Sleep', 
        '9-11 hours': 'High Sleep', '8-9 hours': 'High Sleep', '1-6 hours': 'Low Sleep', '35-36 hours': 'Extremely High Sleep', 
        '40-45 hours': 'Extremely High Sleep', '45-48 hours': 'Extremely High Sleep', '49 hours': 'Extremely High Sleep', 
        '55-66 hours': 'Extremely High Sleep', 'Sleep_Duration': pd.NA, 'Work_Study_Hours': pd.NA, 'No': 'No Sleep', 
        'Unhealthy': 'Low Sleep', 'Pune': pd.NA, 'Indore': pd.NA, 'Moderate': 'Moderate Sleep', '9-5 hours': 'Moderate Sleep', 
        '9-5': 'Moderate Sleep', 'than 5 hours': 'Moderate Sleep', '10-6 hours': 'Moderate Sleep', '3-6 hours': 'Low Sleep', '45': pd.NA
    }
    df['Sleep Duration'] = df['Sleep Duration'].map(sleep_mapping)
    return df
   


 df = mapear_duracion_sueno(df)

    valores_unicos = df['Sleep Duration'].unique()
    print(f"Valores únicos en la columna '{'Sleep Duration'}': {valores_unicos}")

# Ejemplo de filtro de hábitos alimenticios válidos
def filtrar_habitos_alimenticios(df):
    valid_dietary_habits = ['Healthy', 'Unhealthy', 'Moderate', 'Indoor']
    # Asignar 'Unknown' a los hábitos alimenticios no válidos
    df['Dietary Habits'] = df['Dietary Habits'].apply(lambda x: x if x in valid_dietary_habits else 'Unknown')
    return df

df = filtrar_habitos_alimenticios(df)

    valores_unicos = df['Dietary Habits'].unique()
    print(f"Valores únicos en la columna '{'Dietary Habits'}': {valores_unicos}")

# Ejemplo de conversión binaria de columnas
def conversion_binaria(df):
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'No': 0, 'Yes': 1})
    df['Family History of Mental Illness'] = df['Family History of Mental Illness'].map({'No': 0, 'Yes': 1})
    #df['Depression'] = df['Depression'].map({'No': 0, 'Yes': 1})
    return df


df = conversion_binaria(df)

df.head()

from sklearn.preprocessing import OrdinalEncoder
# Ejemplo de codificación ordinal de duración del sueño y hábitos alimenticios
def codificar_ordinal_sueno_habitos(df):
    dietary_habits_order = ['Healthy', 'Moderate', 'Unhealthy', 'Indoor']
    sleep_duration_order = ['No Sleep', 'Very Low Sleep', 'Low Sleep', 'Moderate Sleep', 'High Sleep', 'Extremely High Sleep']
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, categories=[dietary_habits_order, sleep_duration_order])
    df[['Dietary Habits', 'Sleep Duration']] = ordinal_encoder.fit_transform(df[['Dietary Habits', 'Sleep Duration']])
    return df

df = codificar_ordinal_sueno_habitos(df)

df.head()

# Eliminación de columnas innecesarias
def eliminar_columnas(df):
    df_seleccionado = df.drop(['id', 'Name', 'City', 'Working Professional or Student', 'Profession', 'Degree'], axis=1)
    return df_seleccionado.copy()


df_transformado = eliminar_columnas(df)

df_transformado.info()


# Crear un diccionario para almacenar los valores únicos de cada columna
valores_unicos = {col: df[col].unique() for col in df.columns}

# Mostrar los resultados
for columna, valores in valores_unicos.items():
    print(f"Valores únicos en '{columna}': {valores}")

from sklearn.model_selection import train_test_split

X = df_transformado.drop(columns=['Depression'])
y = df_transformado['Depression']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(verbose=100)
}

best_model = None
best_accuracy = 0
model_results = {}

for name, model in models.items():
    
    print(f"\n💻 Training {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred)

    y_train_pred = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    
    print(f'{name} Training Accuracy: {accuracy_train:.4f}')
    print(f'{name} Validation Accuracy: {accuracy_val:.4f}')
    print(classification_report(y_val, y_pred))
    

    model_results[name] = {
        'model': model,
        'train_accuracy': accuracy_train,
        'val_accuracy': accuracy_val,
        'classification_report': classification_report(y_val, y_pred, output_dict=True)
    }
    
   
    if accuracy_val > best_accuracy:
        best_accuracy = accuracy_val
        best_model = model
        
# Visualizamos el mejor modelo
print(f'\n🏆 Mejor Modelo: {best_model.__class__.__name__} with validation accuracy: {best_accuracy:.4f}')

import plotly.express as px
import pandas as pd

# Crear DataFrame con resultados
results_df = pd.DataFrame({
    "Model": list(model_results.keys()),
    "Training Accuracy": [result['train_accuracy'] for result in model_results.values()],
    "Validation Accuracy": [result['val_accuracy'] for result in model_results.values()]
})

# Generar gráficos individuales para cada modelo
for index, row in results_df.iterrows():
    model_name = row["Model"]
    data = pd.DataFrame({
        "Accuracy Type": ["Training Accuracy", "Validation Accuracy"],
        "Value": [row["Training Accuracy"], row["Validation Accuracy"]]
    })
    
    # Generar gráfico
    fig = px.bar(
        data,
        x="Accuracy Type",
        y="Value",
        color="Accuracy Type",
        title=f"Accuracies for {model_name}",
        color_discrete_map={"Training Accuracy": "blue", "Validation Accuracy": "orange"}  # Colores alineados
    )
    
    # Ajustar diseño para barras más delgadas y alineadas
    fig.update_traces(width=0.4)  # Ajustar el ancho de las barras
    fig.update_layout(
        xaxis=dict(title="Accuracy Type", tickmode="linear", tickangle=0),
        yaxis=dict(title="Accuracy", range=[0, 1]),
        bargap=0.2,  # Espacio entre las barras
        title=dict(x=0.5)  # Centrar el título
    )
    
    # Mostrar gráfico
    fig.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Generar Curva ROC para cada modelo
def plot_roc_curve(model, X_val, y_val, model_name):
    # Obtener predicciones de probabilidad
    y_pred_prob = model.predict_proba(X_val)[:, 1]

    # Calcular la Curva ROC y el AUC
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Crear la gráfica
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='orange', label=f'Curva ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')  # Línea diagonal
    plt.title(f'ROC/AUC - {model_name}')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

# Llamar a la función para cada modelo
for model_name, result in model_results.items():
    model = result['model']
    plot_roc_curve(model, X_val, y_val, model_name)

# Importancias de características
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names, model_name):
    try:
        # Extraer importancia de características del modelo
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):  # Para modelos lineales, aunque no se usan aquí
            importances = model.coef_[0]
        else:
            print(f"⚠️ {model_name} no proporciona importancias de características.")
            return

        # Crear DataFrame para ordenarlas
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Visualizar
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.gca().invert_yaxis()
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.show()
    except Exception as e:
        print(f"Error al calcular las importancias de {model_name}: {e}")

# Obtener nombres de las características
feature_names = X.columns

# Generar gráficos para cada modelo con importancia de características
for name, result in model_results.items():
    plot_feature_importance(result['model'], feature_names, name)

import pandas as pd
import matplotlib.pyplot as plt

# Crear un DataFrame con los resultados de los modelos
comparison_data = []
for name, result in model_results.items():
    val_report = result['classification_report']['weighted avg']
    comparison_data.append({
        'Model': name,
        'Train Accuracy': result['train_accuracy'],
        'Validation Accuracy': result['val_accuracy'],
        'Precision': val_report['precision'],
        'Recall': val_report['recall'],
        'F1-Score': val_report['f1-score']
    })

comparison_df = pd.DataFrame(comparison_data)

# Mostrar la tabla en consola
print("Model Comparison Results")
print(comparison_df)

# Visualizar los resultados
def plot_comparison(df, metric, title):
    plt.figure(figsize=(10, 6))
    plt.bar(df['Model'], df[metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.show()

# Gráficos para cada métrica
plot_comparison(comparison_df, 'Train Accuracy', 'Training Accuracy Comparison')
plot_comparison(comparison_df, 'Validation Accuracy', 'Validation Accuracy Comparison')
plot_comparison(comparison_df, 'Precision', 'Precision Comparison')
plot_comparison(comparison_df, 'Recall', 'Recall Comparison')
plot_comparison(comparison_df, 'F1-Score', 'F1-Score Comparison')

import seaborn as sns
import matplotlib.pyplot as plt

# Preparar los datos para Seaborn
metrics_melted = metrics_df.reset_index().melt(id_vars='Model', var_name='Metric', value_name='Value')

# Crear la gráfica de barras
plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_melted, x='Metric', y='Value', hue='Model', palette="tab10")

# Configurar la gráfica
plt.title('Comparación de Métricas entre Modelos', fontsize=16)
plt.ylabel('Valor', fontsize=12)
plt.xlabel('Métrica', fontsize=12)
plt.ylim(0.75, 1.0)  # Ajustar el rango del eje Y
plt.legend(title='Modelos', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Mostrar la gráfica
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Crear el DataFrame con las métricas de cada modelo
comparison_data = []
for name, result in model_results.items():
    val_report = result['classification_report']['weighted avg']
    comparison_data.append({
        'Model': name,
        'Precision': val_report['precision'],
        'Recall': val_report['recall'],
        'F1-Score': val_report['f1-score'],
        'AUC': None  # Placeholder para agregar AUC más adelante
    })

# Calcular el AUC para cada modelo
from sklearn.metrics import roc_auc_score

for idx, (name, result) in enumerate(model_results.items()):
    model = result['model']
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_prob)
    comparison_data[idx]['AUC'] = auc_score

# Convertir a DataFrame
metrics_df = pd.DataFrame(comparison_data)
metrics_df.set_index('Model', inplace=True)

# Preparar datos para el radar
categories = list(metrics_df.columns)
num_vars = len(categories)

# Crear el diagrama de radar
plt.figure(figsize=(8, 8))
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Cerrar el gráfico

for idx, model in enumerate(metrics_df.index):
    values = metrics_df.loc[model].tolist()
    values += values[:1]  # Cerrar el gráfico
    plt.polar(angles, values, label=model, linewidth=2)
    plt.fill(angles, values, alpha=0.25)

plt.xticks(angles[:-1], categories, fontsize=10)
plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0], ["0.8", "0.85", "0.9", "0.95", "1.0"], color="grey", size=7)
plt.ylim(0.75, 1.0)
plt.title("Diagrama de Radar Comparativo", size=14, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

import plotly.express as px
results_df = pd.DataFrame({
    "Model": list(model_results.keys()),
    "Training Accuracy": [result['train_accuracy'] for result in model_results.values()],
    "Validation Accuracy": [result['val_accuracy'] for result in model_results.values()]
})
fig = px.bar(results_df, x="Model", y=["Training Accuracy", "Validation Accuracy"], barmode="group", title="Model Accuracies")
fig.show()

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score

# Definir el número de pliegues
n_splits = 5  # Cambia este valor según sea necesario

# Métrica de evaluación
scoring = 'accuracy'  # Cambia a otras métricas como 'f1', 'precision', etc., si es necesario

# Resultados de Cross-Validation
cv_results = {}

print("\n📊 Cross-Validation Results:")
for name, model in models.items():
    print(f"\n💻 Running Cross-Validation for {name}...")
    
    # Calcular scores de CV
    scores = cross_val_score(model, X, y, cv=n_splits, scoring=scoring)
    mean_score = scores.mean()
    std_score = scores.std()

    # Guardar resultados
    cv_results[name] = {
        'Mean Accuracy': mean_score,
        'Standard Deviation': std_score
    }
    
    # Mostrar resultados
    print(f'{name} - Mean Accuracy: {mean_score:.4f}, Standard Deviation: {std_score:.4f}')

# Crear un DataFrame con los resultados de CV
cv_df = pd.DataFrame(cv_results).T
cv_df.reset_index(inplace=True)
cv_df.rename(columns={'index': 'Model'}, inplace=True)

# Mostrar tabla de resultados
print("\nCross-Validation Summary:")
print(cv_df)

# Graficar los resultados de CV
plt.figure(figsize=(10, 6))
plt.bar(cv_df['Model'], cv_df['Mean Accuracy'], yerr=cv_df['Standard Deviation'], capsize=5)
plt.title('Cross-Validation Accuracy Comparison')
plt.ylabel('Mean Accuracy')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Datos de precisión promedio y desviación estándar
models = cv_df['Model']
mean_accuracy = cv_df['Mean Accuracy']
std_dev = cv_df['Standard Deviation']

# Gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(models, mean_accuracy, yerr=std_dev, capsize=5, color=['blue', 'orange', 'green', 'red'])
plt.title('Precisión Promedio de Validación Cruzada con Desviación Estándar', fontsize=14)
plt.ylabel('Precisión Promedio', fontsize=12)
plt.xlabel('Modelos', fontsize=12)
plt.ylim(0.93, 0.94)  # Ajustar rango si es necesario
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Mostrar gráfico
plt.show()

# Datos de comparación (precisiones obtenidas previamente sin CV)
accuracy_without_cv = {
    'Random Forest': 0.9344,
    'XGBoost': 0.9364,
    'LightGBM': 0.9381,
    'CatBoost': 0.9377
}

# Precisión promedio con CV
accuracy_with_cv = dict(zip(cv_df['Model'], cv_df['Mean Accuracy']))

# Preparar datos
models = list(accuracy_without_cv.keys())
accuracy_without = [accuracy_without_cv[model] for model in models]
accuracy_with = [accuracy_with_cv[model] for model in models]

# Gráfico comparativo
plt.figure(figsize=(10, 6))
x = np.arange(len(models))
plt.plot(x, accuracy_without, marker='o', label='Sin Validación Cruzada', linestyle='--', color='gray')
plt.plot(x, accuracy_with, marker='o', label='Con Validación Cruzada', linestyle='-', color='blue')
plt.title('Comparación de Precisión Promedio con y sin Validación Cruzada', fontsize=14)
plt.ylabel('Precisión Promedio', fontsize=12)
plt.xlabel('Modelos', fontsize=12)
plt.xticks(x, models, rotation=45)
plt.ylim(0.93, 0.94)  # Ajustar rango si es necesario
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Mostrar gráfico
plt.show()

