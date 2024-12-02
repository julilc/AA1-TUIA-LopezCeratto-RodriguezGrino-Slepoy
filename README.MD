
##### Uso mediante Docker
Para descargar la imagen:
docker pull julilc/austr_rain_app:v7

Para correr la imagen:
docker run --rm -v "<carpeta_contenedora_de_tu_csv_en_tu_pc>:/app/data" -v "<carpeta_donde_se_guarde_el_csv_salida_en_tu_pc>:/app/output" julilc/austr_rain_app:v7 '/app/inferencia.py' '/app/data/weather_sample.csv'

Subimos y utilizamos el archivo weather_samples dentro de la carpeta /src

##### Uso del archivo descargado
Ejecute:
python inferencia.py

##### Pasar parámetros
El parámetro debe ser la ubicación absoluta o relativa del archivo .csv que desea predecir y debe contener las columnas:
Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday


Ejemplo:

"src\weather_sample.csv"

##### Errores
En caso de que algún dato no esté pasado correctamente, el programa le devolverá dónde se encontró un error y será necesario ejecutarlo
nuevamente para ingresar datos de forma correcta.
