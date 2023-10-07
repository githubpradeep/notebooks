# filename: game_of_life_v3.py

import pygame
import numpy as np

# Pygame initialization
pygame.init()

# Window size
width, height = 800, 800

# Creating the screen
screen = pygame.display.set_mode((height, width))

# Color of the cells
bg = 25, 25, 25
alive_color = 255, 255, 255

# Number of cells in each direction
nxC, nyC = 80, 80

dimCW = width / nxC
dimCH = height / nyC

# State of the cells. Alive = 1; Dead = 0
# Initialize the state randomly
state = np.random.randint(0, 2, (nxC, nyC))

# Game loop
running = True
while running:

    new_state = np.copy(state)

    screen.fill(bg)

    for y in range(0, nxC):
        for x in range(0, nyC):

            # Calculate the number of nearby cells
            n_neigh = state[(x - 1) % nxC, (y - 1) % nyC] + \
                      state[(x) % nxC, (y - 1) % nyC] + \
                      state[(x + 1) % nxC, (y - 1) % nyC] + \
                      state[(x - 1) % nxC, (y) % nyC] + \
                      state[(x + 1) % nxC, (y) % nyC] + \
                      state[(x - 1) % nxC, (y + 1) % nyC] + \
                      state[(x) % nxC, (y + 1) % nyC] + \
                      state[(x + 1) % nxC, (y + 1) % nyC]

            # Rule 1: A dead cell with exactly 3 alive neighbours, "revives"
            if state[x, y] == 0 and n_neigh == 3:
                new_state[x, y] = 1

            # Rule 2: An alive cell with less than 2 or more than 3 alive neighbours, "dies"
            elif state[x, y] == 1 and (n_neigh < 2 or n_neigh > 3):
                new_state[x, y] = 0

            # Creating the polygon for each cell to draw
            poly = [(x * dimCW, y * dimCH),
                    ((x + 1) * dimCW, y * dimCH),
                    ((x + 1) * dimCW, (y + 1) * dimCH),
                    (x * dimCW, (y + 1) * dimCH)]

            # And draw the cell for each x, y
            if new_state[x, y] == 0:
                pygame.draw.polygon(screen, (128, 128, 128), poly, 1)
            else:
                pygame.draw.polygon(screen, alive_color, poly, 0)

    # Update the state of the game
    state = np.copy(new_state)

    # Update the screen
    pygame.display.flip()

    # Check for QUIT event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()