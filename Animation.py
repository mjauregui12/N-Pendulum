import sys
import numpy as np
from numpy import cos, sin, pi
import pygame
from Calculations import RK4_step

#All the code necessary to run the animation
def update(theta):
    scale = 50
    X = [l[0] * sin(theta[0])]
    Y = [l[0] * cos(theta[0])]
    if len(theta) > 1:
        for i in range(len(theta)-1):
            X.append(X[i]+l[i+1]*sin(theta[i+1]))
            Y.append(Y[i]+l[i+1]*cos(theta[i+1]))

    X = np.array(X)
    Y = np.array(Y)
    return X+offset[0], Y+offset[1]


def render(X, Y):
    scale = 10

    pygame.draw.circle(screen, BLACK, offset, 6)
    pygame.draw.line(screen, BLACK, offset, (X[0], Y[0]), 5)
    pygame.draw.circle(screen, BLACK, (X[0], Y[0]), int(m[0]*scale))
    if len(X) > 1:
        pygame.draw.line(screen, BLACK, (X[0], Y[0]), (X[1], Y[1]), 5)
        for k in range(len(X)-1):
            pygame.draw.circle(screen, BLACK, (X[k], Y[k]), int(m[k]*scale))
            pygame.draw.line(screen, BLACK, (X[k], Y[k]), (X[k+1], Y[k+1]), 5)
    pygame.draw.circle(screen, BLACK, (X[len(X)-1], Y[len(X)-1]), int(m[len(X)-1])*scale)

#Initializes window
w,h = 1500, 750
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
trajectory = []
offset = (750, 50)

screen = pygame.display.set_mode((w, h))
screen.fill(WHITE)
pygame.display.update()
clock = pygame.time.Clock()

#Values to be adjusted
N = 4 #Recommended for N < 20
FPS = 60
t = 0.0
delta_t = 1/FPS
m = np.ones(N)
l = np.ones(N)*50
y = np.append(np.zeros(N) + pi/3, np.zeros(N))
g = 9.81

#Animation
while True:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    X, Y = update(y[0:N])
    render(X, Y)

    t += delta_t
    y = y + RK4_step(y, t, delta_t)

    clock.tick(FPS)
    pygame.display.update()