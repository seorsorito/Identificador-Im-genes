import tensorflow as tf
from keras import layers, models

from sklearn.model_selection import train_test_split
import os

num_classes=1

# Directorio donde se encuentran las imágenes organizadas por clases
data_dir = 'C://Users//victor//Desktop//Filtrador//imagenes'

# Obtener una lista de todas las clases de motores
classes = os.listdir(data_dir)

# Diccionario para almacenar las rutas de las imágenes por clase
image_paths = {cls: [os.path.join(data_dir, cls, img) for img in os.listdir(os.path.join(data_dir, cls))] 
               for cls in classes}

# Listas para almacenar las rutas de las imágenes y las etiquetas
image_list = []
labels = []

# Recorrer las clases y las rutas de las imágenes para agregarlas a las listas
for cls, imgs in image_paths.items():
    image_list.extend(imgs)
    labels.extend([cls] * len(imgs))

# Dividir el conjunto de datos en conjuntos de entrenamiento, validación y prueba
# Usaremos 70% para entrenamiento, 15% para validación y 15% para prueba
train_images, test_val_images, train_labels, test_val_labels = train_test_split(image_list, labels, test_size=0.3, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(test_val_images, test_val_labels, test_size=0.5, random_state=42)

# Imprimir el tamaño de cada conjunto
print("Número de imágenes de entrenamiento:", len(train_images))
print("Número de imágenes de validación:", len(val_images))
print("Número de imágenes de prueba:", len(test_images))


#------------------------------------------------------------------------------------------------------------------------------------

# Definir el modelo de red neuronal convolucional
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # num_classes es el número de clases de motores
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Accuracy del modelo en el conjunto de prueba: {test_acc}')

# Guardar el modelo
model.save('motor_classifier.h5')
