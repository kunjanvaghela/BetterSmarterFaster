# Import and initialize the pygame library
import pygame
import math
from pygame.locals import (
    KEYDOWN,
)
pygame.init()

# Define constants for the screen width and height
screen_width = 1050
screen_height = 970

def draw_nodes():
    # equation : (x-h)^2 + (y-k)^2 = r^2
    h = screen_width/2
    k = screen_height/2
    r = 450

    atRadius = 0

    for i in range(50):
        x =  r * math.cos(atRadius)
        y = r * math.sin(atRadius)
        pygame.draw.circle(screen, (0, 150, 255), (x+h, y+k), 20)
        # pygame.display.flip()
        atRadius += 0.125664
    pass

# Set up the drawing window
screen = pygame.display.set_mode([screen_width, screen_height])

# Run until the user asks to quit
running = True
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill((255, 255, 255))

    # Draw a solid blue circle in the center
    # pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)
    draw_nodes()

    # Flip the display
    pygame.display.flip()


# Done! Time to quit.
pygame.quit()