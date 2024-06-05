# Configuración de la simulación
ANCHO_PANTALLA = 800
ALTO_PANTALLA = 600
FPS = 60

# Configuración de los semáforos
NUM_SEMAFOROS = 4
DURACION_VERDE_BASE = 10  # Duración base del verde en segundos
DURACION_AMARILLO = 3  # Duración del amarillo en segundos

# Configuración de los carriles
CARRILES = [
    ((100, 200), (700, 200), (1, 0)),  # Carril 1: (inicio_x, inicio_y), (fin_x, fin_y), (dirección_x, dirección_y)
    ((700, 400), (100, 400), (-1, 0)),  # Carril 2: (inicio_x, inicio_y), (fin_x, fin_y), (dirección_x, dirección_y)
    ((400, 100), (400, 500), (0, 1)),  # Carril 3: (inicio_x, inicio_y), (fin_x, fin_y), (dirección_x, dirección_y)
    ((400, 500), (400, 100), (0, -1))  # Carril 4: (inicio_x, inicio_y), (fin_x, fin_y), (dirección_x, dirección_y)
]

# Configuración de los vehículos
VELOCIDAD_VEHICULO = 100  # Velocidad de los vehículos en píxeles por segundo

# Configuración de la detección de vehículos
REGION_DETECCION = (350, 250, 450, 350)  # (x_min, y_min, x_max, y_max)
UMBRAL_DETECCION = 0.5  # Umbral de confianza para la detección de vehículos

# Configuración de los colores
COLOR_FONDO = (0, 0, 0)  # Negro
COLOR_CARRIL = (255, 255, 255)  # Blanco
COLOR_SEMAFORO_ROJO = (255, 0, 0)  # Rojo
COLOR_SEMAFORO_AMARILLO = (255, 255, 0)  # Amarillo
COLOR_SEMAFORO_VERDE = (0, 255, 0)  # Verde
COLOR_VEHICULO = (0, 0, 255)  # Azul

# Configuración de las rutas de archivos
RUTA_MODELO_DETECCION = "ruta/al/modelo/de/deteccion.h5"
RUTA_CLASES_DETECCION = "ruta/al/archivo/de/clases.txt"
