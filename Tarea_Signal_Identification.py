import cv2
import numpy as np

def detectar_forma(contorno):
    
    peri = cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, 0.04 * peri, True)
    vertices = len(approx)

    # Clasificar la forma según el número de vértices
    if vertices == 3:
        sorted_vertices = sorted(approx, key=lambda x: x[0][1])
        if sorted_vertices[0][0][1] < sorted_vertices[1][0][1] and sorted_vertices[0][0][1] < sorted_vertices[2][0][1]:
            return "Triángulo Normal"
        else:
            return "Triángulo Invertido"
    elif vertices == 8:
        return "Octágono"
    elif vertices > 8:
        return "Círculo"
    else:
        return "Indefinido"

def detectar_senales(imagen, referencias):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Detectar características SIFT en la imagen de la cámara
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gris, None)

    if des is None or len(des) == 0:
        return imagen, None

    # Configuración del match
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    num_coincidencias = []

# Encontrar coincidencias entre los camara y la imagen   
    for kp_ref, des_ref in referencias:
        matches = flann.knnMatch(des_ref, des, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        num_coincidencias.append(len(good_matches))

    indice_max_coincidencias = np.argmax(num_coincidencias)

    # Si hay suficientes coincidencias, devolver el índice de la señal detectada
    if num_coincidencias[indice_max_coincidencias] >= 10:
        return imagen, indice_max_coincidencias
    else:
        return imagen, None

# Lista de imagenes de referencias
nombres_archivos = ["stop", "giveaway", "straigth", "turnaround", "turnleft", "turnright", "workinprogress"]


referencias = []

# Iterar sobre los nombres de archivos y cargar las imágenes de referencia
for nombre in nombres_archivos:
    
    ruta_imagen = f"{nombre}.jpg"

    ref_img = cv2.imread(ruta_imagen, 0)

    # Verificar si la imagen se ha leído correctamente
    if ref_img is None:
        print(f"No se pudo leer la imagen : {ruta_imagen}")
        continue

    # Detectar características SIFT en la imagen de referencia
    sift = cv2.SIFT_create()
    ref_kp, ref_des = sift.detectAndCompute(ref_img, None)

    # Verificar si se detectaron características
    if ref_des is None:
        print(f"No se encontró referencia: {ruta_imagen}")
        continue

    # Almacenar los puntos clave 
    referencias.append((ref_kp, ref_des))

# Verificar si se han cargado correctamente las imágenes de referencia
if len(referencias) == 0:
    print("No se han cargado correctamente las imágenes de referencia")
else:
    print("Se han cargado correctamente las imágenes de referencia")


cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar la referencia en la imagen de la camara
    frame, indice_senal_detectada = detectar_senales(frame, referencias)

    # Si se detectó una señal mostrar el nombre 
    if indice_senal_detectada is not None:
        nombres_senales = ["Stop", "Giveaway", "Straigth", "Turnaround", "Turn Left", "Turn Right", "Work in Progress"]
        nombre_senal = nombres_senales[indice_senal_detectada]
        cv2.putText(frame, nombre_senal, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.imshow('Detector de Señales de Tránsito', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
