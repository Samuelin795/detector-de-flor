import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos Iris y entrenar el modelo
iris = load_iris()
X = iris.data
y = iris.target

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo y entrenarlo
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Función para capturar una imagen de la cámara y mostrarla
def capture_image():
    # Abrir la cámara
    cap = cv2.VideoCapture(0)  # 0 es el índice de la cámara predeterminada

    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return None

    print("Captura de imagen en progreso...")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer el frame.")
            break

        # Mostrar la imagenq
        cv2.imshow("Captura de Imagen", frame)

        # Esperar por una tecla (presionar 'q' para salir)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Liberar la cámara y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()

    return frame

# Capturar una imagen
captured_image = capture_image()

# Mostrar la imagen capturada
if captured_image is not None:
    cv2.imwrite("captured_flower.jpg", captured_image)  # Guardar la imagen
    print("Imagen capturada y guardada como 'captured_flower.jpg'.")
