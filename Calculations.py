import numpy as np
from numpy import cos, sin, sum, pi
from numpy.linalg import inv

#Utilizes a Runge Kutta process in order to solve the equations of motion
def G(Y, t):
    n = int(len(Y)/2)
    theta = np.array(Y[0:n])
    vel = np.array(Y[n:2*n])
    M = fill_matrixM(theta, n)
    F = fill_matrixF(theta, vel, n)
    accel = inv(M).dot(F.T)

    return np.append(vel, accel)


def fill_matrixM(theta, n):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i][j] = l[i]*l[j]*cos(theta[i]-theta[j])*sum(m[max(i, j):n])
    return M


def fill_matrixF(theta, v, n):
    F = np.zeros(n)
    for i in range(n):
        F[i] = -1.0*summation_1(i, theta, v, n) - g*l[i]*sin(theta[i])*sum(m[i:n])
    return F


def summation_1(index, theta, v, n):
    val = 0
    for k in range(n):
        val += sum(m[max(index, k):n])*l[index]*l[k]*sin(theta[index]-theta[k])*v[k]*v[k]
    return val


def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y+0.5*k1*dt, t+0.5*dt)
    k3 = G(y+0.5*k2*dt, t+0.5*dt)
    k4 = G(y+k3*dt, t+dt)
    return dt*(k1 + 2*k2 + 2*k3 + k4)/6

#default
N = 10
m = np.ones(N)
l = np.ones(N)
y = np.append(np.zeros(N) + pi/2, np.zeros(N))
g = 9.81
