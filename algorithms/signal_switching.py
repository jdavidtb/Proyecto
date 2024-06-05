import numpy as np

class ControladorSemaforos:
    def __init__(self, num_semaforos, duracion_verde_base, duracion_amarillo):
        self.num_semaforos = num_semaforos
        self.duracion_verde_base = duracion_verde_base
        self.duracion_amarillo = duracion_amarillo
        self.tiempos_verde = np.zeros(num_semaforos)
        self.estado_semaforos = np.zeros(num_semaforos, dtype=int)

    def actualizar_tiempos_verde(self, conteos_vehiculos):
        total_vehiculos = np.sum(conteos_vehiculos)
        if total_vehiculos > 0:
            proporciones_vehiculos = conteos_vehiculos / total_vehiculos
            self.tiempos_verde = proporciones_vehiculos * self.duracion_verde_base
        else:
            self.tiempos_verde = np.ones(self.num_semaforos) * (self.duracion_verde_base / self.num_semaforos)

    def cambiar_semaforo(self, semaforo_actual):
        self.estado_semaforos[semaforo_actual] = 0  # Rojo
        semaforo_siguiente = (semaforo_actual + 1) % self.num_semaforos
        self.estado_semaforos[semaforo_siguiente] = 2  # Verde
        
        # Amarillo para el semáforo actual
        self.estado_semaforos[semaforo_actual] = 1
        duracion_amarillo = self.duracion_amarillo
        
        return semaforo_siguiente, duracion_amarillo, self.tiempos_verde[semaforo_siguiente]

    def ejecutar_ciclo_semaforos(self, conteos_vehiculos):
        self.actualizar_tiempos_verde(conteos_vehiculos)
        semaforo_actual = 0
        
        while True:
            semaforo_siguiente, duracion_amarillo, duracion_verde = self.cambiar_semaforo(semaforo_actual)
            
            # Esperar la duración del amarillo
            yield semaforo_actual, duracion_amarillo, "amarillo"
            
            # Esperar la duración del verde
            yield semaforo_siguiente, duracion_verde, "verde"
            
            semaforo_actual = semaforo_siguiente
