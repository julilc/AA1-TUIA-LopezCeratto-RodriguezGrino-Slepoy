
##### Uso mediante Docker
Para descargar la imagen:
docker pull julilc/austr_rain_app:v2

Para correr la imagen:
docker run -it --rm <usuario>/austr_rain_app:v2

##### Uso del archivo descargado
Ejecute:
python inferencia.py

##### Pasar parámetros
Los datos deben ser pasados separados por una ",". El orden es:
'Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am','WindDir3pm','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm','Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am','Temp3pm', 'RainToday'

Ejemplo:

Ingrese los datos separados por ',': 2009-01-01,Williamtown,15.2,34.0,0.0,11.6,9.2,NW,41.0,WNW,SE,17.,20.0,62.0,36.0,1005.2,1001.5,3.0,6.0,23.7,28.7,No

##### Errores
En caso de que algún dato no esté pasado correctamente, el programa le devolverá dónde se encontró un error y será necesario ejecutarlo
nuevamente para ingresar datos de forma correcta.