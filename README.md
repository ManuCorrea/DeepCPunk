# DeepCPunk

Este proyecto tiene como finalidad transformar imágenes de ciudades(o cualquier otro tipo) a una estética cyberpunk.
Para cumplir este objetivo usamos CycleGAN.

![](images_notebook/Gen.jpg)

## Notebook
Como presentacion y para facilitar el aprendizaje he hecho un notebook completamente en castellano y orientado a principiantes.

Para visualizarlo necesitaréis [instalar Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html). Posteriormente ejecutar el notebook CycleGAN.ipynb

### Instalación 🔧

Para poder entrenar y usar esta red deberemos instalar varios paquetes de Python, todos incluidos en requeriments.txt

Dentro del directorio deberemos ejecutar.

```
pip install -r requirements.txt
```
# Pruebas

### Modo fácil
Si no tienes mucha idea de Python, tu PC no es potente o simplemente te encuentras en un día vago te propongo lo siguiente:
* Ve a Twitter
* Pon un twit con la imágen que quieras transformar a CyberPunk
* Mencióname [@_ManuCorrea\_](https://twitter.com/_ManuCorrea_) y pon #DeepCPunk
* Si todo va bien te responderá con la imágen ya procesada por una red con los pesos variantes entre las distintas opciones.

Dada las limitaciones y que uso la API pública(sólo puedo acceder a las últimas 20 menciones) no puedo asegurar estabilidad en el servicio.
Esta opción estará disponible temporalmente y cuando se anuncien los elegidos (tanto si salgo elegido como si no).

### Prueba con imágenes
No necesitas un gran equipo para probarlo de este manera.
* Introduce las imágenes que desees en el directorio ./prueba/input_imgs/imgs
* En el script podrás seleccionar distintos pesos generados. Cada número corresponde a la epoch en la que se generaron.
* Ejecuta el script dentro del ./prueba/
```
python3 process_img.py
```
* Las imágenes generadas se guardarán en ./prueba/

### Prueba con webcam
Para probar en directo la distopía cyberpunk creada deberás contar con una webcam y un equipo decente en cuanto a 
procesador y RAM. Si tienes CUDA podrás visualizarlo a unos FPS decentes. 

```
python3 feedWeb.py -e 60
```
-e elige los pesos para cargar en la demo

### Dataset
Para obtener y usar el dataset deberás descargar 
[este zip](https://drive.google.com/file/d/1xry9VYzKhAcP2Dbzg9yf72Bl5Lgvufo0/view?usp=sharing) en la carpeta ./datasets/

### Entrenamiento

Para realizar el entrenamiento recomiendo tener CUDA instalado.

Puedes obtener resultados con pocas etapas de entrenamiento, los primeros resultados son muy
exagerados. Conforme avanzan las etapas los resultados se suavizan más. 

¡Eres libre de experimentar y coger el resultado
que más te guste!
```
python3 train.py
```

### Agradecimientos
* [Jun-Yan Zhu](https://github.com/junyanz) por crear CycleGAN y compartir código y conocimiento.
* [Dot CSV](https://www.youtube.com/channel/UCy5znSnfMsDwaLlROnZ7Qbg) por la oportunidad de participar en la competición.
* Toda la comunidad OpenSource por compartir herramientas y conocimientos.