# Usar una imagen base de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios al contenedor
COPY inferencia.py rnn.h5 rqst.py imputers.csv modifiers.py scalers.pkl requirements.txt /app/

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Definir el comando predeterminado
CMD ["python", "inferencia.py"]
