import pygame
import numpy as np
from algorithms.signal_switching import ControladorSemaforos

# Colores
NEGRO = (0, 0, 0)
BLANCO = (255, 255, 255)
ROJO = (255, 0, 0)
VERDE = (0, 255, 0)
AMARILLO = (255, 255, 0)

class Simulacion:
    def __init__(self, ancho, alto, num_semaforos, duracion_verde_base, duracion_amarillo):
        self.ancho = ancho
        self.alto = alto
        self.pantalla = pygame.display.set_mode((ancho, alto))
        pygame.display.set_caption("Simulación de Tráfico")
        
        self.controlador_semaforos = ControladorSemaforos(num_semaforos, duracion_verde_base, duracion_amarillo)
        
        self.semaforos = []
        self.carriles = []
        self.vehiculos = []
        
        self.reloj = pygame.time.Clock()
    
    def dibujar_semaforos(self):
        for semaforo in self.semaforos:
            semaforo.dibujar(self.pantalla)
    
    def dibujar_carriles(self):
        for carril in self.carriles:
            carril.dibujar(self.pantalla)
    
    def dibujar_vehiculos(self):
        for vehiculo in self.vehiculos:
            vehiculo.dibujar(self.pantalla)
    
    def actualizar_vehiculos(self, dt):
        for vehiculo in self.vehiculos:
            vehiculo.actualizar(dt)
    
    def ejecutar(self):
        corriendo = True
        while corriendo:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    corriendo = False
            
            self.pantalla.fill(NEGRO)
            
            self.dibujar_carriles()
            self.dibujar_semaforos()
            self.dibujar_vehiculos()
            
            pygame.display.flip()
            
            dt = self.reloj.tick(60) / 1000
            self.actualizar_vehiculos(dt)
        
        pygame.quit()

class Semaforo:
    def __init__(self, x, y, estado_inicial=0):
        self.x = x
        self.y = y
        self.estado = estado_inicial
    
    def dibujar(self, pantalla):
        if self.estado == 0:  # Rojo
            color = ROJO
        elif self.estado == 1:  # Amarillo
            color = AMARILLO
        else:  # Verde
            color = VERDE
        
        pygame.draw.circle(pantalla, color, (self.x, self.y), 20)

class Carril:
    def __init__(self, inicio, fin, direccion):
        self.inicio = inicio
        self.fin = fin
        self.direccion = direccion
    
    def dibujar(self, pantalla):
        pygame.draw.line(pantalla, BLANCO, self.inicio, self.fin, 2)

class Vehiculo:
    def __init__(self, x, y, velocidad, direccion):
        self.x = x
        self.y = y
        self.velocidad = velocidad
        self.direccion = direccion
    
    def actualizar(self, dt):
        self.x += self.velocidad * dt * self.direccion[0]
        self.y += self.velocidad * dt * self.direccion[1]
    
    def dibujar(self, pantalla):
        pygame.draw.rect(pantalla, BLANCO, (self.x, self.y, 10, 10))
