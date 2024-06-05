import pygame
from simulation.traffic_simulation import Simulacion
from simulation.config import ANCHO_PANTALLA, ALTO_PANTALLA, NUM_SEMAFOROS, DURACION_VERDE_BASE, DURACION_AMARILLO, CARRILES, VELOCIDAD_VEHICULO
from Models.object_detection import detectar_vehiculos
from algorithms.signal_switching import ControladorSemaforos

def main():
    pygame.init()

    # Crear la simulación de tráfico
    simulacion = Simulacion(ANCHO_PANTALLA, ALTO_PANTALLA, NUM_SEMAFOROS, DURACION_VERDE_BASE, DURACION_AMARILLO)

    # Crear los carriles
    for inicio, fin, direccion in CARRILES:
        simulacion.agregar_carril(inicio, fin, direccion)

    # Crear los semáforos
    for i in range(NUM_SEMAFOROS):
        x = ANCHO_PANTALLA // 2
        y = ALTO_PANTALLA // (NUM_SEMAFOROS + 1) * (i + 1)
        simulacion.agregar_semaforo(x, y)

    # Controlador de semáforos
    controlador_semaforos = ControladorSemaforos(NUM_SEMAFOROS, DURACION_VERDE_BASE, DURACION_AMARILLO)

    # Bucle principal de la simulación
    reloj = pygame.time.Clock()
    while True:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                return

        # Detectar vehículos en la región de interés
        imagen_simulacion = pygame.surfarray.array3d(pygame.display.get_surface())
        conteos_vehiculos = detectar_vehiculos(imagen_simulacion)

        # Actualizar los tiempos de los semáforos según los conteos de vehículos
        controlador_semaforos.actualizar_tiempos_verde(conteos_vehiculos)

        # Obtener el estado actual de los semáforos
        estado_semaforos = controlador_semaforos.obtener_estado_semaforos()

        # Actualizar el estado de los semáforos en la simulación
        for i, semaforo in enumerate(simulacion.semaforos):
            semaforo.estado = estado_semaforos[i]

        # Actualizar la simulación
        simulacion.actualizar(reloj.tick(60) / 1000)

        # Dibujar la simulación
        simulacion.dibujar()

        pygame.display.flip()

if __name__ == "__main__":
    main()