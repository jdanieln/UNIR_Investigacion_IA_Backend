import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt




def manejo_valores_faltantes(df):
    faltantes = df.isnull().sum()
    print(faltantes)
    df.dropna(how='all')
    df['Quantity Ordered'] = pd.to_numeric(df['Quantity Ordered'], errors='coerce')
    df = df.dropna(how='all')
    df.info()
    num_rows_before = df.shape[0]
    df.dropna(inplace=True)
    num_rows_after = df.shape[0]
    num_rows_deleted = num_rows_before - num_rows_after
    print("Número de filas eliminadas:", num_rows_deleted)    
    return df

def seleccionar_columnas(df, columnas):
    return df[columnas]

def convert_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M')
    except ValueError:
        return pd.to_datetime(date_str, format='%m/%d/%y %H:%M')

def cargar_datos(nombre_archivo):
    
    df = pd.read_csv(nombre_archivo)
    df.info()    
    return df

def tratamiento_columnas(df):

  df['Order Date'] = df['Order Date'].apply(convert_date)
  df['Year'] = df['Order Date'].dt.year
  df['Month'] = df['Order Date'].dt.month
  df['Day'] = df['Order Date'].dt.day
  df['Hour'] = df['Order Date'].dt.hour
  df['Minute'] = df['Order Date'].dt.minute
  print("ULTIMO")
  df.info()
  return df

def manejo_duplicados(df):
    duplicados = df.duplicated()
    df.drop_duplicates(inplace=True)
    return df

def obtener_datos_descriptivos(df):
  print(df.describe())
  print(df.describe(include=['O']))
  unique_products = df['Product'].unique()
  print(unique_products)
  media_cantidad = df['Quantity Ordered'].mean()
  mediana_cantidad = df['Quantity Ordered'].median()
  desviacion_estandar_cantidad = df['Quantity Ordered'].std()
  resumen_cantidad = df['Quantity Ordered'].describe()

def top_10_productos_mas_vendidos(df,mostrar_grafico,mostrar_resumen):
  top_products = df.groupby('Product')['Quantity Ordered'].sum().nlargest(10)

  # Obtén el producto más vendido
  producto_mas_vendido = top_products.idxmax()
  cantidad_mas_vendida = top_products.max()
  resumen = "El gráfico muestra los 10 productos más vendidos.\n"   
  resumen=f"El producto más vendido fue '{producto_mas_vendido}' con una cantidad de {cantidad_mas_vendida} unidades vendidas."
 
  
  if mostrar_grafico:
    sns.barplot(x=top_products.index, y=top_products.values)
    plt.xticks(rotation=90)
    plt.xlabel('Producto')
    plt.ylabel('Cantidad Vendida')
    plt.title('Productos Más Vendidos por Cantidad Vendida')
  if mostrar_resumen:
    print(resumen)
  return resumen 

def frecuencia_de_ventas_por_producto(df,mostrar_grafico,mostrar_resumen):
  product_frequency = df['Product'].value_counts()
  plt.figure(figsize=(12, 6))
  product_frequency.plot(kind='bar', color='blue')
  producto_mas_frecuente = product_frequency.idxmax()
  frecuencia_maxima = product_frequency.max()
  resumen = "El gráfico muestra la frecuencia de ventas de los productos:.\\n"
  resumen += f"El producto más frecuente es '{producto_mas_frecuente}' con una frecuencia de {frecuencia_maxima} compras."
  if mostrar_grafico:
    plt.xlabel('Producto')
    plt.ylabel('Frecuencia')
    plt.title('Análisis de Frecuencia de Productos')
    plt.xticks(rotation=90)
    plt.grid(axis='y')    
    plt.show()
  if mostrar_resumen:
    print(resumen)
  return resumen

def distribucion_de_la_cantidad_vendida_de_productos(df, mostrar_grafico, mostrar_resumen):
    # Calcular el valor más frecuente (moda)
    moda = df['Quantity Ordered'].mode().values[0] 
    resumen = "El gráfico la distribución de la cantidad de productos vendidos en función de la frecuencia de ventas.\n"
    resumen += "La cantidad más frecuente vendida es"+ str(moda)+ "unidades."     
    if mostrar_grafico:
      plt.hist(df['Quantity Ordered'], bins=20, edgecolor='k')
      plt.xlabel('Cantidad Vendida')
      plt.ylabel('Frecuencia')
      plt.title('Distribución de Cantidad Vendida')        
      plt.show()

    if mostrar_resumen:
      print(resumen)
    return resumen 
def ventas_por_mes_y_dia_de_la_semana(df,mostrar_grafico,mostrar_resumen):
  df['Month'] = df['Order Date'].dt.month
  df['Day of Week'] = df['Order Date'].dt.dayofweek
  heatmap_data = df.pivot_table(index='Month', columns='Day of Week', values='Quantity Ordered', aggfunc='sum')
  mes_max_ventas = heatmap_data.sum(axis=1).idxmax()
  dia_max_ventas = heatmap_data.sum().idxmax()
  max_ventas = heatmap_data.loc[mes_max_ventas, dia_max_ventas]
  mes_min_ventas = heatmap_data.sum(axis=1).idxmin()
  dia_min_ventas = heatmap_data.sum().idxmin()
  min_ventas = heatmap_data.loc[mes_min_ventas, dia_min_ventas]
  total_ventas = heatmap_data.sum().sum()
  promedio_ventas_diario = heatmap_data.mean().mean()
  resumen = "En el análisis del mapa de calor de ventas por mes y día de la semana, se encontraron las siguientes estadísticas:\n"
  resumen += "El mes con la mayor cantidad de ventas es " + str(mes_max_ventas) + ", con un total de " + str(max_ventas) + " ventas.\n"
  resumen += "El día de la semana con la mayor cantidad de ventas es " + str(dia_max_ventas) + ".\n"
  resumen += "El día de la semana con la menor cantidad de ventas es " + str(dia_min_ventas) + "."
  resumen += "El total de ventas en el conjunto de datos es de " + str(total_ventas) + " unidades."
  resumen += "El promedio de ventas por día de la semana es de aproximadamente " + "{:.2f}".format(promedio_ventas_diario) + " unidades."

  if mostrar_grafico:
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
def ventas_por_hora_del_dia_y_de_la_semana(df,mostrar_grafico,mostrar_resumen):
  df['Hour'] = df['Order Date'].dt.hour
  df['Day of Week'] = df['Order Date'].dt.dayofweek 

  heatmap_data =df.pivot_table(index='Hour', columns='Day of Week', values='Quantity Ordered', aggfunc='sum')
  dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

  hora_pico_total = heatmap_data.idxmax().values
  dia_pico_total = heatmap_data.idxmax(axis=1).idxmax()  
  resumen = "Resumen del Mapa de Calor de Cantidad Vendida por Hora del Día y Día de la Semana:"
  resumen += "El mapa de calor muestra las horas del día y los días de la semana con las mayores ventas."
  resumen += "En general, la hora de mayor cantidad vendida es a las" + str(hora_pico_total)+", y el día de la semana con la mayor cantidad vendida es el día "+ str(dia_pico_total) +"."

  if mostrar_grafico:
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



def dispersion_cantidad_vendida_por_producto(df,mostrar_grafico,mostrar_resumen):
  resumen = "Resumen del Gráfico de Dispersión de Cantidad Vendida por Producto:"
  resumen += "Este gráfico muestra la relación entre la cantidad vendida y los productos."

  if mostrar_grafico:
    sns.scatterplot(x='Product', y='Quantity Ordered', data=df)
    plt.xticks(rotation=90)
    plt.xlabel('Producto')
    plt.ylabel('Cantidad Vendida')
    plt.title('Gráfico de Dispersión de Cantidad Vendida por Producto')      
    plt.grid(True)
    plt.show()
  if mostrar_resumen:
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
