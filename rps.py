# filename: rps_simulator.py

import pygame
import random
import os

# Define some colors
WHITE = (200, 200, 200)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)  # Define a green color

# Define the properties of our rock, paper, scissors particles
PARTICLE_RADIUS = 10
IMAGE_SIZE = (25, 25)  # New size for the images

PARTICLE_COLORS = {
    "rock": pygame.transform.scale(pygame.image.load(os.path.join(os.getcwd(), 'rock.png')), IMAGE_SIZE),
    "paper": pygame.transform.scale(pygame.image.load(os.path.join(os.getcwd(), 'paper.png')), IMAGE_SIZE),
    "scissors": pygame.transform.scale(pygame.image.load(os.path.join(os.getcwd(), 'scissors.png')), IMAGE_SIZE)
}

# Define the rules of the game
RULES = {
    "rock": "scissors",
    "scissors": "paper",
    "paper": "rock"
}

class Particle:
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)

    def draw(self, screen):
        screen.blit(PARTICLE_COLORS[self.type], (self.x, self.y))

    def move(self):
        self.x += self.vx
        self.y += self.vy

        # Bounce off the edges of the screen
        if self.x < PARTICLE_RADIUS or self.x > size[0] - PARTICLE_RADIUS:
            self.vx = -self.vx
        if self.y < PARTICLE_RADIUS or self.y > size[1] - PARTICLE_RADIUS:
            self.vy = -self.vy

    def check_collision(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        distance = (dx**2 + dy**2)**0.5
        return distance < 2*PARTICLE_RADIUS

    def react(self, other):
        if RULES[self.type] == other.type and other in particles:
            particles.remove(other)
        elif RULES[other.type] == self.type and self in particles:
            particles.remove(self)

# Initialize Pygame
pygame.init()

# Set the width and height of the screen [width, height]
size = (900, 600)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Rock Paper Scissors Simulator")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# Create a list to store our particles
particles = []


# Create some particles
for i in range(300):
    x = random.randint(PARTICLE_RADIUS, size[0] - PARTICLE_RADIUS)
    y = random.randint(PARTICLE_RADIUS, size[1] - PARTICLE_RADIUS)
    type = random.choice(["rock", "paper", "scissors"])
    particles.append(Particle(x, y, type))

# -------- Main Program Loop -----------
while not done:
    # --- Main event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # --- Game logic should go here
    for particle in particles:
        particle.move()
        for other in particles:
            if particle.check_collision(other):
                particle.react(other)

    # --- Drawing code should go here
    screen.fill(WHITE)
    for particle in particles:
        particle.draw(screen)

    # --- Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    # --- Limit to 60 frames per second
    clock.tick(60)

# Close the window and quit.
pygame.quit()