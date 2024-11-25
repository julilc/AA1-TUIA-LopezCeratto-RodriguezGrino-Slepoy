import pickle
import sys
import numpy as np

# Cargar modelo y transformadores
with open('rnn.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

with open('imputers_scalers.pkl', 'rb') as transform_file:
    transformadores = pickle.load(transform_file)

def inferir(datos_entrada):
    # Asegurarse de que los datos sean un array numpy
    datos_np = np.array(datos_entrada).reshape(1, -1)
    
    # Aplicar imputers y scalers
    for transformador in transformadores:
        datos_np = transformador.transform(datos_np)
    
    # Realizar predicción
    prediccion = modelo.predict(datos_np)
    return prediccion

if __name__ == '__main__':
    # Leer datos de entrada desde la línea de comandos
    datos = list(map(float, sys.argv[1:]))
    resultado = inferir(datos)
    print(f"Predicción: {resultado}")