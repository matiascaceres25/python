import cv2 

# se inserta las rutas a los archivos del modelo pre-entrenado y el archivo de configuración asi  mismo
prototxt = "model/MobileNetSSD_deploy.prototxt.txt"
model = "model/MobileNetSSD_deploy.caffemodel"

# Clases de objetos que el modelo puede detectar(las variables del modelo entrenado)
classes = {
    0: "fondo", 1:"avion", 2:"bicicleta",3:"pajaro", 4:"barco",5:"botella", 6:"colectivo",7:"auto", 8:"gato",9:"silla", 10:"vaca",
    11:"comedor", 12:"perro",13:"caballo", 14:"moto",15:"persona", 16:"pantera",17:"oveja", 18:"sofa",19:"tren", 20:"tvmonitor"
}

# se carga el modelo
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# se carga la imagen
image = cv2.imread("imagenes/jaja.jpg")

#  las dimensiones de la imagen
height, width, _ = image.shape

# se configura la imagen a 300x300 (tamaño esperado por el modelo)
image_resized = cv2.resize( image, (300,300))

# linea para procesar la imagen para que coincida con el formato requerido por el modelo
blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300,300), (127.5, 127.5, 127.5))

print("blob.shape", blob.shape)

# se establece la entrada del modelo
net.setInput(blob)

# ejecutar la detección de objetos
detections = net.forward()

# iterar sobre las detecciones
for detection in detections[0][0]:
    # se imprime la detección actual
    print(detection)
    
    # Si la confianza en la detección es mayor a 0.45
    if detection[2] > 0.45:
        # se obtine la etiqueta del objeto detectado
        label = classes[detection[1]]
        print("Label:", label)
        
        # se obtiene las coordenadas de la caja delimitadora
        box = detection[3:7] * [width, height, width, height]
        x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # este linea dibuja la caja delimitadora y etiqueta sobre la imagen
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (8, 75, 252), 2 )
        cv2.putText(image, label, (x_start, y_start), 1, 1.2, (255,0,0), 2)

# se meustra la imagen con las detecciones
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
