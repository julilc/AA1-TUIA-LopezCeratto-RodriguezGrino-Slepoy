
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


if __name__ == '__main__':
    # Leer datos de entrada desde la línea de comandos

    from rqst import *
    from modifiers import *
    datos = input("Ingrese los datos: ").split(",")
    datos =pd.DataFrame([datos], columns = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm', 'RainToday'])
    # Cargar modelo y transformadores
    print(datos)
    rnn_rain = load_model("Docker/rnn.h5", custom_objects={"CustomRecallMetric": CustomRecallMetric})

    mediana_moda = pd.read_csv('Docker/imputers.csv')

    with open('Docker/scalers.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    prob_lluvia = inferir(datos)
    if prob_lluvia <0.5:
        resultado= 'No llueve'
    else:
        resultado = 'Llueve'
    print(f"Predicción: {resultado}, probabilidades de lluvia: {prob_lluvia}")