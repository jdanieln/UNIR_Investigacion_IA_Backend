from datetime import datetime, timedelta
import json
from flask import Flask, request
from flask import jsonify
from config import config
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from funciones import *


def create_app(env):
    app = Flask(__name__)
    CORS(app)
    app.config['DEBUG'] = True
    app.config.from_object(env)

    return app


env = config['development']
app = create_app(env)
def convert_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y %H:%M')
    except ValueError:
        return pd.to_datetime(date_str, format='%m/%d/%y %H:%M')

@app.route('/upload', methods=['POST'])
def upload_csv():
    global df
    df = cargar_datos('https://raw.githubusercontent.com/jdanieln/UNIR_Investigacion_IA_Backend/main/data/data.csv')    
    return 'CSV en la solicitud.', 200

flag_csv=False
@app.route("/dataframe")
def dataFrame():
    df = cargar_datos('https://raw.githubusercontent.com/jdanieln/UNIR_Investigacion_IA_Backend/main/data/data.csv')    
    selected_columns = ["Product", "Quantity Ordered", "Order Date"]
    df = seleccionar_columnas(df,selected_columns)
    df = manejo_valores_faltantes(df)
    dfToFront=df.copy()
    df = tratamiento_columnas(df)

    # Convertir el DataFrame a JSON
    df_json = dfToFront.to_json(orient='records')
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

@app.route("/")
def root():

    return "Works!!"


@app.route("/getPrediction/<string:start_date>/<string:end_date>/<string:product>", methods=["GET"])
def get_sales_by_date(start_date, end_date):
    results=""
    return jsonify(status=True, data=results), 200


if __name__ == '__main__':
    app.run( host='0.0.0.0',port=5002)
 