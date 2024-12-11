
def inferir(datos_rain):
    # Asegurarse de que los datos sean un array numpy
    
    datos_rain['Date'] = pd.to_datetime(datos_rain['Date'])
    datos_rain['Day_of_Year'] = datos_rain['Date'].dt.dayofyear
    datos_rain['Month'] = datos_rain['Date'].dt.month
    datos_rain['Season'] = datos_rain['Month'].apply(obtener_estacion)

    datos_rain = datos_rain.apply(lambda fila: llenar_faltantes_por_mes(fila, mediana_moda), axis=1)

    datos_rain.drop(columns= ['Month', 'Season'], inplace = True) 
    
    datos_rain = codificacion(datos_rain)
    
    datos_rain.drop(columns = ['Date'], inplace = True)
    
    datos_rain = scaler.transform(datos_rain)


    # Realizar predicción
    prediccion = rnn_rain.predict(datos_rain)

    return prediccion


def cheq(datos):
    """
    Esta función chequea los datos para asegurarse de que estén bien ingresados por el usuario.
    """
    errores = []
    ciudades = ['Williamtown', 'MountGinini', 'Bendigo', 'Portland', 'Watsonia', 'Dartmoor', 'Townsville', 'Launceston', 'AliceSprings', 'Katherine']
    coordenadas = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    cols_mayor_a_0 = ['Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
                      'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm']
    cols_coordenadas = ['WindDir9am', 'WindDir3pm']
    contador = 1
    col_temps = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']

    indices = range(len(datos))
    
    for i in indices:
        # Validar fecha
        if pd.isna(datos['Date'][i]):
            errores.append(f"{contador}. Error: Fecha nula.")
            contador += 1

        # Validar ubicación
        if datos['Location'].iloc[i] not in ciudades and not pd.isna(datos['Location'][i]):
            errores.append(f"{contador}. Error: Ubicación desconocida. Debe ser una de: {ciudades}.")
            contador += 1

        # Validar temperaturas mínimas y máximas
        for temp in col_temps:
            if pd.isna(datos[temp][i]):
                continue
            try:
                datos[temp][i] = float(datos[temp][i])
                if not (-15 <= datos[temp].iloc[0] <= 50):
                    errores.append(f"{contador}. Error: en fila {i}, columna {temp} fuera de rango (-15 a 50).")
                    contador += 1
            except ValueError:
                errores.append(f"{contador}. Error: en fila {i}, columna {temp} no es un número válido.")
                contador += 1

        # Validar columnas mayores a 0
        for col in cols_mayor_a_0:
            if pd.isna(datos[col][i]):
                continue
            try:
                datos[col][i] = float(datos[col][i])
                if datos[col].iloc[i] < 0:
                    errores.append(f"{contador}. Error: en fila {i}, columna {col} debe ser mayor o igual a 0.")
                    contador += 1
            except ValueError:
                errores.append(f"{contador}. Error: en fila {i}, columna {col} no es un número válido.")
                contador += 1

        # Validar coordenadas
        for col in cols_coordenadas:
            if pd.isna(datos[col][i]):
                continue
            if datos[col].iloc[i] not in coordenadas:
                errores.append(f"{contador}. Error: en fila {i}, columna {col} no es una coordenada válida. Debe estar entre: {coordenadas}.")
                contador += 1

    if contador>1:
        return False, "\n".join(errores)
    else:
        return True, "Datos correctos."
     



if __name__ == '__main__':
    # Leer datos de entrada desde la línea de comandos
    from rqst import *
    from modifiers import *
    if len(sys.argv) < 2:
        print("Error: Debes proporcionar un archivo .csv como argumento.")
        sys.exit(1)

    datos = sys.argv[1]
    print(f"Procesando el archivo: {datos}")    
    try:
        datos = pd.read_csv(datos)
    except ValueError:
        print('Archivo con formato incorrecto o no se encontró el archivo')
    
    output_file = "/app/output/predicciones.csv"  # Ruta dentro del contenedor
    cols_num = ['MinTemp','MaxTemp','Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
                      'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
 
    # for col in cols_num:
    #     # Asegurarse de que el valor sea numérico o "NaN"
    #     datos[col] = datos[col].apply(lambda x: pd.to_numeric(x, errors='coerce') if isinstance(x, (int, float)) else (x if x != 'NaN' else pd.NA))

    # for col in datos.columns:
    #     print(datos[col])
    salida, mensaje = cheq(datos)

    if salida is False:
        print('Se encontraron los siguientes errores')
        print(mensaje)
        
    else:
        # Cargar modelo y transformadores
        rnn_rain = load_model("rnn.h5", custom_objects={"CustomRecallMetric": CustomRecallMetric})

        mediana_moda = pd.read_csv('imputers.csv')
        mediana_moda.set_index('Month', inplace= True)

        with open('scalers.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        prob_lluvia = inferir(datos)
        resultados = []
        if len(prob_lluvia)>1:
            for i in range(len(prob_lluvia)):
                
                if prob_lluvia[i] <rnn_rain:
                    resultado= 'No llueve'
                else:
                    resultado = 'Llueve'
                resultados.append({
                'Fila': i+1,
                'Prediccion': resultado,
                'Probabilidad de lluvia': prob_lluvia[i]})
        else:
            if prob_lluvia <rnn_rain.umbral:
                resultado= 'No llueve'
            else:
                resultado = 'Llueve'
            resultados.append({
                'Fila': 1,
                'Prediccion': resultado,
                'Probabilidad de lluvia': prob_lluvia})
            
        df_salida = pd.DataFrame(resultados)
        df_salida.to_csv(output_file, index=False)
        print(f"Archivo de resultados guardado en {output_file}")

        