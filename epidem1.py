# filename: epidemic_simulator.py

import pygame
import random

# Set up some constants
WIDTH, HEIGHT = 800, 600
DOT_SIZE = 5
DOTS = 200
SPEED = 2
INFECTED_COLOR = (255, 0, 0)
HEALTHY_COLOR = (0, 255, 0)
RECOVERED_COLOR = (0, 0, 255)
FRAMES_TO_RECOVER = 500

class Dot:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.vx = random.randint(-SPEED, SPEED)
        self.vy = random.randint(-SPEED, SPEED)
        self.infected = False
        self.frames_infected = 0

    def move(self):
        self.x += self.vx
        self.y += self.vy

        if self.x < 0 or self.x > WIDTH:
            self.vx = -self.vx
        if self.y < 0 or self.y > HEIGHT:
            self.vy = -self.vy

    def draw(self, screen):
        if self.infected:
            color = INFECTED_COLOR
        elif self.frames_infected > 0:
            color = RECOVERED_COLOR
        else:
            color = HEALTHY_COLOR
        pygame.draw.circle(screen, color, (self.x, self.y), DOT_SIZE)

    def check_infection(self, others):
        if self.infected:
            for dot in others:
                if dot.frames_infected == 0 and abs(self.x - dot.x) < DOT_SIZE and abs(self.y - dot.y) < DOT_SIZE:
                    dot.infected = True
            self.frames_infected += 1
            if self.frames_infected > FRAMES_TO_RECOVER:
                self.infected = False

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    dots = [Dot() for _ in range(DOTS)]
    dots[0].infected = True  # Start with one infected dot

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEMOTION:
                break
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        for dot in dots:
            dot.move()
            dot.draw(screen)
            dot.check_infection(dots)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()