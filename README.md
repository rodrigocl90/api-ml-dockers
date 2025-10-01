## Instalación y ejecución


git clone https://github.com/rodrigocl/api-ml-docker.git 

cd api-ml-docker/

docker build -t ml-flask-api:1.0 . 

docker run -d -p 5000:5000 ml-flask-api:1.0

API de Machine Learning que ha sido dockerizada, construida y actualmente está funcionando y lista para recibir solicitudes en el puerto 8080

![API de Machine Learning que ha sido dockerizada, construida y actualmente está funcionando y lista para recibir solicitudes en el puerto 8080](/imagenes/imagen1.png)

Este endpoint se utiliza para verificar el estado de la aplicación. Sirve como check de salud (health check) para confirmar que el servicio está corriendo correctamente.

![Texto alternativo](/imagenes/imagen2.png)

Este endpoint se utiliza para ver información del modelo 

![Texto alternativo](/imagenes/imagen3.png)

Prueba de envío de predicciones mediante POST

![Prueba de envío de predicciones mediante POST](/imagenes/imagen4.png)
