import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt




# Función para manejar valores faltantes en un DataFrame
def manejo_valores_faltantes(df):
    # Calcula la cantidad de valores nulos en cada columna
    faltantes = df.isnull().sum()
    print(faltantes)
    
    # Elimina filas con todos los valores nulos
    df.dropna(how='all')
    
    # Convierte la columna 'Quantity Ordered' a valores numéricos y elimina filas con valores nulos
    df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
    df = df.dropna(how='all')
    
    # Muestra información sobre el DataFrame
    df.info()
    
    # Calcula el número de filas eliminadas
    num_rows_before = df.shape[0]
    df.dropna(inplace=True)
    num_rows_after = df.shape[0]
    num_rows_deleted = num_rows_before - num_rows_after
    print("Número de filas eliminadas:", num_rows_deleted)
    
    return df
# Función para seleccionar columnas específicas de un DataFrame
def seleccionar_columnas(df, columnas):
    return df[columnas]

# Función para convertir una cadena de fecha en un objeto de fecha y hora
def convert_date(date_str):
    try:
        # Intenta convertir la fecha con formato 'MM/DD/YYYY HH:MM'
        return pd.to_datetime(date_str, format='%m/%d/%y %H:%M')
        
    except ValueError:
        # Si falla, intenta con formato 'MM/DD/YY HH:MM'
          return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M')
def convert_date_str(date_str):
    try:
        # Intenta convertir la fecha con formato 'MM/DD/YYYY HH:MM'
        return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
        
    except ValueError:
        # Si falla, intenta con formato 'MM/DD/YY HH:MM'
          return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S')
# Función para cargar datos desde un archivo CSV
def cargar_datos(nombre_archivo):
    df = pd.read_csv(nombre_archivo)
    df.info()
    return df

# Función para realizar el tratamiento de columnas relacionadas con fechas
def tratamiento_columnas(df):
    
    
    df['Order Date'] = df['Order Date'].apply(convert_date)
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Day'] = df['Order Date'].dt.day
    df['Hour'] = df['Order Date'].dt.hour
    df['Minute'] = df['Order Date'].dt.minute
    df['Day of Week'] = df['Order Date'].dt.dayofweek
    # Muestra información sobre el DataFrame después del tratamiento
    df.info()
    
    return df

# Función para manejar registros duplicados en un DataFrame
def manejo_duplicados(df):
    # Busca y elimina registros duplicados
    duplicados = df.duplicated()
    df.drop_duplicates(inplace=True)
    
    return df


# Función para obtener estadísticas descriptivas de un DataFrame
def obtener_datos_descriptivos(df):
    # Muestra estadísticas descriptivas para columnas numéricas
    print(df.describe())
    
    # Muestra estadísticas descriptivas para columnas categóricas
    print(df.describe(include=['O']))
    
    # Obtiene los productos únicos en la columna 'Product'
    unique_products = df['Product'].unique()
    print(unique_products)
    
    # Calcula la media, mediana y desviación estándar de la cantidad vendida
    media_cantidad = df['Quantity Ordered'].mean()
    mediana_cantidad = df['Quantity Ordered'].median()
    desviacion_estandar_cantidad = df['Quantity Ordered'].std()
    resumen_cantidad = df['Quantity Ordered'].describe()

# Función para identificar y visualizar los 10 productos más vendidos
def top_10_productos_mas_vendidos(df, mostrar_grafico, mostrar_resumen):
    # Agrupa los productos y suma las cantidades vendidas, luego selecciona los 10 productos con mayores ventas
    top_products = df.groupby('Product')['Quantity Ordered'].sum().nlargest(10)

    # Obtiene el producto más vendido y la cantidad más vendida
    producto_mas_vendido = top_products.idxmax()
    cantidad_mas_vendida = top_products.max()

    # Crea un resumen
    resumen = "El gráfico muestra los 10 productos más vendidos.\n"
    resumen += f"El producto más vendido fue '{producto_mas_vendido}' con una cantidad de {cantidad_mas_vendida} unidades vendidas."

    if mostrar_grafico:
        # Muestra un gráfico de barras de los productos más vendidos
        sns.barplot(x=top_products.index, y=top_products.values)
        plt.xticks(rotation=90)
        plt.xlabel('Producto')
        plt.ylabel('Cantidad Vendida')
        plt.title('Productos Más Vendidos por Cantidad Vendida')

    if mostrar_resumen:
        print(resumen)

    return resumen

# Función para analizar la frecuencia de ventas por producto
def frecuencia_de_ventas_por_producto(df, mostrar_grafico, mostrar_resumen):
    # Calcula la frecuencia de ventas de cada producto
    product_frequency = df['Product'].value_counts()

    if mostrar_grafico:
        # Muestra un gráfico de barras de la frecuencia de ventas
        plt.figure(figsize=(12, 6))
        product_frequency.plot(kind='bar', color='blue')
        plt.xlabel('Producto')
        plt.ylabel('Frecuencia')
        plt.title('Análisis de Frecuencia de Productos')
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.show()

    # Obtiene el producto más frecuente y su frecuencia máxima
    producto_mas_frecuente = product_frequency.idxmax()
    frecuencia_maxima = product_frequency.max()

    # Crea un resumen
    resumen = "El gráfico muestra la frecuencia de ventas de los productos:\n"
    resumen += f"El producto más frecuente es '{producto_mas_frecuente}' con una frecuencia de {frecuencia_maxima} compras."

    if mostrar_resumen:
        print(resumen)

    return resumen

# Función para analizar la distribución de la cantidad vendida de productos
def distribucion_de_la_cantidad_vendida_de_productos(df, mostrar_grafico, mostrar_resumen):
    # Calcula la moda (valor más frecuente) de la cantidad vendida
    moda = df['Quantity Ordered'].mode().values[0]

    if mostrar_grafico:
        # Muestra un histograma de la cantidad vendida
        plt.hist(df['Quantity Ordered'], bins=20, edgecolor='k')
        plt.xlabel('Cantidad Vendida')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Cantidad Vendida')
        plt.show()

    # Crea un resumen
    resumen = "El gráfico muestra la distribución de la cantidad de productos vendidos en función de la frecuencia de ventas.\n"
    resumen += f"La cantidad más frecuente vendida es {moda} unidades."

    if mostrar_resumen:
        print(resumen)

    return resumen

# Función para analizar las ventas por mes y día de la semana
def ventas_por_mes_y_dia_de_la_semana(df, mostrar_grafico, mostrar_resumen):
    

    # Crea una tabla de calor (heatmap) que muestra la cantidad vendida en función del mes y el día de la semana
    heatmap_data = df.pivot_table(index='Month', columns='Day of Week', values='Quantity Ordered', aggfunc='sum')
    
    # Encuentra el mes con las mayores ventas y el día de la semana con las mayores ventas
    mes_max_ventas = heatmap_data.sum(axis=1).idxmax()
    dia_max_ventas = heatmap_data.sum().idxmax()
    
    # Obtiene el valor máximo de ventas
    max_ventas = heatmap_data.loc[mes_max_ventas, dia_max_ventas]
    
    # Encuentra el mes con las menores ventas y el día de la semana con las menores ventas
    mes_min_ventas = heatmap_data.sum(axis=1).idxmin()
    dia_min_ventas = heatmap_data.sum().idxmin()
    
    # Obtiene el valor mínimo de ventas
    min_ventas = heatmap_data.loc[mes_min_ventas, dia_min_ventas]
    
    # Calcula el total de ventas en el conjunto de datos
    total_ventas = heatmap_data.sum().sum()
    
    # Calcula el promedio de ventas por día de la semana
    promedio_ventas_diario = heatmap_data.mean().mean()
    
    # Crea un resumen de los hallazgos
    resumen = "En el análisis del mapa de calor de ventas por mes y día de la semana, se encontraron las siguientes estadísticas:\n"
    resumen += f"El mes con la mayor cantidad de ventas es {mes_max_ventas}, con un total de {max_ventas} ventas.\n"
    resumen += f"El día de la semana con la mayor cantidad de ventas es {dia_max_ventas}.\n"
    resumen += f"El día de la semana con la menor cantidad de ventas es {dia_min_ventas}.\n"
    resumen += f"El total de ventas en el conjunto de datos es de {total_ventas} unidades.\n"
    resumen += f"El promedio de ventas por día de la semana es de aproximadamente {promedio_ventas_diario:.2f} unidades."

    if mostrar_grafico:
        # Muestra un mapa de calor de las ventas
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=.5)
        plt.title('Mapa de Calor de Ventas por Mes y Día de la Semana')
        plt.xlabel('Día de la Semana')
        plt.ylabel('Mes')
        plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6], labels=['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])
        plt.yticks(rotation=0)
        plt.show()
    
    if mostrar_resumen:
        print(resumen)
    
    return resumen


def ventas_por_hora_del_dia_y_de_la_semana(df, mostrar_grafico, mostrar_resumen):
    # Crea una tabla de calor (heatmap) que muestra la cantidad vendida en función de la hora del día y el día de la semana
    heatmap_data = df.pivot_table(index='Hour', columns='Day of Week', values='Quantity Ordered', aggfunc='sum')
    
    # Define los nombres de los días de la semana
    dias_semana = [ "Domingo","Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"]
    
    # Encuentra la hora de pico total y el día de la semana de pico total
    hora_pico_total = heatmap_data.idxmax().values
    dia_pico_total = heatmap_data.idxmax(axis=1).idxmax() 
      # Encuentra el día de la semana pico total
    dia_semana_pico_total = heatmap_data.loc[hora_pico_total, :].idxmax()
    print("dia_pico_total")
    print(dia_pico_total)
    # Crea un resumen de los hallazgos
    resumen = "Resumen del Mapa de Calor de Cantidad Vendida por Hora del Día y Día de la Semana:\n"
    resumen += "El mapa de calor muestra las horas del día y los días de la semana con las mayores ventas.\n"
  

    if mostrar_grafico:
        # Muestra un mapa de calor de las ventas
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=.5)
        plt.title('Mapa de Calor de Cantidad Vendida por Hora del Día y Día de la Semana')
        plt.xlabel('Día de la Semana')
        plt.ylabel('Hora del Día')  
        plt.grid(True)
        plt.show()
    
    if mostrar_resumen:
        print(resumen)
    
    return resumen


# Función para crear un gráfico de dispersión de la cantidad vendida por producto
def dispersion_cantidad_vendida_por_producto(df, mostrar_gráfico, mostrar_resumen):
    # Define el resumen explicativo del gráfico de dispersión
    resumen = "Resumen del Gráfico de Dispersión de Cantidad Vendida por Producto:\n"
    resumen += "Este gráfico muestra la relación entre la cantidad vendida y los productos."

    if mostrar_gráfico:
        # Crea un gráfico de dispersión con los productos en el eje X y la cantidad vendida en el eje Y
        sns.scatterplot(x='Product', y='Quantity Ordered', data=df)
        plt.xticks(rotation=90)
        plt.xlabel('Producto')
        plt.ylabel('Cantidad Vendida')
        plt.title('Gráfico de Dispersión de Cantidad Vendida por Producto')
        plt.grid(True)
        plt.show()

    if mostrar_resumen:
        # Muestra el resumen si se solicita
        print(resumen)

    return resumen

def historial_de_ventas_por_semana(df,mostrar_grafico,mostrar_resumen):
  df['Order Date'] = pd.to_datetime(df['Order Date'])
  df['Year'] = df['Order Date'].dt.year
  df['Week'] = df['Order Date'].dt.isocalendar().week
  weekly_sales = df.groupby(['Year', 'Week'])['Quantity Ordered'].sum()

  resumen = "Resumen del Historial de Ventas por Semana:"
  resumen += "Este gráfico muestra el historial de ventas por semana a lo largo de los años."
  resumen += "Cada barra en el gráfico representa una semana y muestra la cantidad vendida en esa semana."
  resumen += "El eje x muestra las semanas y el eje y muestra la cantidad vendida."
  if mostrar_grafico:
    plt.figure(figsize=(12, 6))
    weekly_sales.plot(kind='bar', color='blue')
    plt.xlabel('Semana')
    plt.ylabel('Cantidad Vendida')
    plt.title('Historial de Ventas por Semana')
    plt.xticks(rotation=90)       
    plt.grid(True)
    plt.show()
  if mostrar_resumen:
    print(resumen)
  return resumen     
