import datetime
import json
from flask import Flask
from flask import jsonify
from requests import HTTPError
from config import config
from flask_cors import CORS
import pandas as pd
import pandas as pd
import numpy as np
from funciones import *
import os
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras


import csv
if not os.path.exists('data.csv'):
    # Si el archivo no existe, crea uno utilizando 
    df = cargar_datos('https://raw.githubusercontent.com/jdanieln/UNIR_Investigacion_IA_Backend/main/data/data.csv')    

    selected_columns = ["Product", "Quantity Ordered", "Order Date"]
    df = seleccionar_columnas(df,selected_columns)
    df = manejo_valores_faltantes(df)
    df = tratamiento_columnas(df)    
    try:
    # Convierte el DataFrame en un archivo CSV
        df.to_csv('data.csv', index=False)
        print(f"Archivo CSV  creado en la misma carpeta del proyecto.")
    except Exception as e:
        print(f"Error al crear el archivo CSV: {e}")

try:
    # Intenta leer el archivo CSV como un DataFrame
    df = pd.read_csv('data.csv')
    print("El archivo CSV ok")
except FileNotFoundError:
    # Si el archivo no se encuentra, realiza otra acción (por ejemplo, muestra un mensaje)

    print("El archivo CSV no se encontró en la misma carpeta del proyecto.")
 

def create_app(env):
    app = Flask(__name__)
    CORS(app)
    app.config['DEBUG'] = env.DEBUG
    app.config.from_object(env)
    app.config['df'] = df
    return app

env = config['development']
app = create_app(env)

flag_csv=False
@app.route("/dataframe")
def dataFrame():

    df = app.config['df'] 
    print(df.head(10))
    df['Order Date'] = df['Order Date'].apply(convert_date_str)
    # Convertir el DataFrame a JSON
    df_json = df.to_json(orient='records')
    resumen_top_10_productos_mas_vendidos=top_10_productos_mas_vendidos(df,False,True)
    resumen_frecuencia_de_ventas_por_producto=frecuencia_de_ventas_por_producto(df,False,True)
    resumen_distribucion_de_la_cantidad_vendida_de_productos=distribucion_de_la_cantidad_vendida_de_productos(df,False,True)
    resumen_ventas_por_mes_y_dia_de_la_semana=ventas_por_mes_y_dia_de_la_semana(df,False,True)    
    resumen_ventas_por_hora_del_dia_y_de_la_semana=ventas_por_hora_del_dia_y_de_la_semana(df,False,True)
    resumen_dispersion_cantidad_vendida_por_producto=dispersion_cantidad_vendida_por_producto(df,False,False) 
    resumen_historial_de_ventas_por_semana=historial_de_ventas_por_semana(df,False,False)        
    resultados_dict = {
        'resumenes': {
            'resumen_top_10_productos_mas_vendidos': resumen_top_10_productos_mas_vendidos,
            'resumen_frecuencia_de_ventas_por_producto': resumen_frecuencia_de_ventas_por_producto,
            'resumen_distribucion_de_la_cantidad_vendida_de_productos': resumen_distribucion_de_la_cantidad_vendida_de_productos,
            'resumen_ventas_por_mes_y_dia_de_la_semana': resumen_ventas_por_mes_y_dia_de_la_semana,
            'resumen_ventas_por_hora_del_dia_y_de_la_semana': resumen_ventas_por_hora_del_dia_y_de_la_semana,
            'resumen_dispersion_cantidad_vendida_por_producto': resumen_dispersion_cantidad_vendida_por_producto,
            'resumen_historial_de_ventas_por_semana': resumen_historial_de_ventas_por_semana             
        }
    }
    resultados_json = json.dumps(resultados_dict)
    return jsonify({"dataframe":df_json,"resumenes":resultados_json})
def root():
    return "Works!!"
@app.route("/test")
def test():
    df = app.config['df']  # Obtener el DataFrame desde la configuración de la aplicación
    # Imprimir los primeros 10 registros del DataFrame
    first_10_records = df.head(10)
    return str(first_10_records)
@app.route("/createCSV")
def createCSV(dataframe, csv_filename):
    try:
        # Convierte el DataFrame en un archivo CSV
        dataframe.to_csv(csv_filename, index=False)
        print(f"Archivo CSV '{csv_filename}' creado en la misma carpeta del proyecto.")
        return "OK", 200
    except Exception as e:
        print(f"Error al crear el archivo CSV: {e}")
        return "Error al crear el archivo CSV", 500
def root():
    return "Works!!"
@app.route("/getPrediction/<string:start_date>/<string:end_date>/<string:product>", methods=["GET"])
def getPrediction(start_date, end_date,product):
    try:
        # Cargar el modelo previamente entrenado
        print("Cargando modelo")  
        loaded_model = load_model("./../modelo-ia/modelo.keras")
        # Preprocesar las fechas y otros datos según sea necesario
        # Aquí asumimos que las fechas se proporcionan en formato "YYYY-MM-DD HH:mm:ss"
        start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")


        print("startDate",start_date)
        print("endDate",end_date)                    
        # Aquí puedes realizar cualquier otro preprocesamiento necesario de los datos de entrada
        # Por ejemplo, codificar la variable categórica "product" si es necesario
        # Crear un ejemplo de input_data (asegúrate de que tenga las mismas características que se utilizaron durante el entrenamiento)
        # En este ejemplo, se asume que tienes un conjunto de características que coincide con las que se utilizaron en el modelo
        input_data = np.array([[
        start_date.year, start_date.month, start_date.day, start_date.hour, start_date.minute, start_date.weekday(),
        product  # Añade la variable "product" codificada aquí si es necesario
        ]])

        # Realizar la predicción
        predictions = loaded_model.predict(input_data)

        # Las predicciones se pueden ajustar según sea necesario antes de incluirlas en la respuesta JSON

        # Convertir las predicciones a una lista de Python
        predictions_list = predictions.tolist()

        # Crear un diccionario que contenga las predicciones
        response = {
        "predictions": predictions_list
        }
        print(response)
        return jsonify(status=True, data=response), 200

    except Exception as e:
        return jsonify(status=False, error=str(e)), 500

if __name__ == '__main__':
    app.run()
 