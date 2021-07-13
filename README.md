# Detección de personas

DNN es un módulo de OpenCV desarrollado para trabajar con redes neuronales profundas, anteriormente este debía se instalado desde *opencv_contrib* pero a partir de la versión 3.3 ya forma parte oficial de la librería por lo que no se requieren dependencias externas.

Para la detección de personas en tiempo real se hace uso de los siguientes modelos pre-entrenados provenientes de los frameworks mas conocidos:

- Inception-SSD ([TensorFlow](https://www.tensorflow.org/))
- MobileNet-SSD ([Caffe](http://caffe.berkeleyvision.org/))
- YOLOv3 ([Darknet](https://pjreddie.com/darknet/))
Descargue el archivo de pesos YOLO v3 previamente entrenado desde este [enlace](https://drive.google.com/file/d/1RoP4rVJo8f2ERZNaTRF_A3hwOUHt3ODd/view?usp=sharing) y colócalo en el directorio YOLOv3.
