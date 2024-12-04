import numpy as np
import cv2
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Función para capturar una imagen de la cámara
def capture_image():
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

        # Mostrar la imagen
        cv2.imshow("Captura de Imagen", frame)

        # Esperar por una tecla (presionar 'q' para salir)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Liberar la cámara y cerrar la ventana
    cap.release()
    cv2.destroyAllWindows()

    return frame

# Función para extraer características de la flor de la imagen
def extract_features_from_image(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro de Canny para detectar bordes
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    # Mostrar los bordes detectados
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Aquí se debería agregar la lógica para extraer las características
    # Para simplificar, devolveremos valores aleatorios como ejemplo
    return np.random.rand(4)  # Características aleatorias (longitud y anchura del sépalo/pétalo)

# Capturar una imagen
captured_image = capture_image()

# Verificar si se capturó la imagen
if captured_image is not None:
    # Extraer características de la imagen
    features = extract_features_from_image(captured_image)

    # Realizar la predicción con el modelo entrenado
    predicted_species = clf.predict([features])

    # Mostrar la predicción
    print(f"La especie de la flor es: {iris.target_names[predicted_species][0]}")

    # Evaluar el modelo con las métricas estándar
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrecisión del modelo: {accuracy:.2f}")
    print("Informe de clasificación:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Visualizar el árbol de decisión
    plt.figure(figsize=(12, 8))
    plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
    plt.show()
