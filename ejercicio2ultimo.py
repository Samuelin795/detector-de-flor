import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# 1. Cargar la imagen
image = cv2.imread('image1.jpg')  # Cambia 'flor.jpg' por la ruta de tu imagen

# 2. Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Aplicar un umbral para segmentar la flor del fondo
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Mostrar la imagen original y la imagen umbralizada
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Imagen Original')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Imagen Umbralizada')
plt.imshow(thresh, cmap='gray')

plt.show()

# 4. Encontrar contornos en la imagen umbralizada
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. Seleccionar el contorno más grande (la flor)
largest_contour = max(contours, key=cv2.contourArea)

# 6. Calcular el área y el perímetro de la flor
area = cv2.contourArea(largest_contour)
perimeter = cv2.arcLength(largest_contour, True)

# 7. Ajustar un rectángulo delimitador para obtener dimensiones aproximadas (altura y ancho)
x, y, w, h = cv2.boundingRect(largest_contour)

# Mostrar el contorno y el rectángulo delimitador
image_with_contour = image.copy()
cv2.drawContours(image_with_contour, [largest_contour], -1, (0, 255, 0), 2)
cv2.rectangle(image_with_contour, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Mostrar los resultados
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image_with_contour, cv2.COLOR_BGR2RGB))
plt.title(f"Área: {area:.2f}, Perímetro: {perimeter:.2f}, Ancho: {w}, Alto: {h}")
plt.show()

# 8. Imprimir características extraídas
print(f"Área: {area:.2f}")
print(f"Perímetro: {perimeter:.2f}")
print(f"Ancho: {w}, Alto: {h}")

# 9. Crear un conjunto de datos de ejemplo con características extraídas de varias imágenes
# Suponiendo que tienes varias flores con etiquetas para entrenamiento
# Aquí es solo un ejemplo con una imagen, agrega más filas para un conjunto real.

features = np.array([[area, perimeter, w, h]])  # Características extraídas
labels = np.array([0])  # Aquí 0 representa la especie de la flor (por ejemplo, Setosa)

# 10. Entrenamiento de un modelo de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)

# Entrenamiento con las características extraídas
clf.fit(features, labels)

# 11. Hacer una predicción con las características extraídas de la nueva imagen
y_pred = clf.predict(features)
print("Predicción de la flor:", y_pred)

# 12. Si tienes más imágenes con sus características, puedes agregarlas al conjunto de datos:
# Por ejemplo, podrías cargar otra imagen, extraer sus características y predecir su clase
