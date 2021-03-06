# Detección de personas

DNN es un módulo de OpenCV desarrollado para trabajar con redes neuronales profundas, anteriormente este debía se instalado desde *opencv_contrib* pero a partir de la versión 3.3 ya forma parte oficial de la librería por lo que no se requieren dependencias externas.

Podremos cargar y utilizar modelos pre-entrenados provenientes de los frameworks mas conocidos como:

- [Caffe](http://caffe.berkeleyvision.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Torch](http://torch.ch/)
- [Darknet](https://pjreddie.com/darknet/)

Para la detección de personas en tiempo real se hace uso de los siguientes:

- Inception-SSD (Tensorflow)
- MobileNet-SSD (Caffe)
- YOLOv3 (Darknet)
