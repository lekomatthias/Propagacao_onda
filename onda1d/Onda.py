from Onda_lib import *

N = 50
t = 1000
L = np.pi
dx = L/N
dt = 0.1*(dx*dx)
lim = [-1.01, 1.01]

pos = np.zeros(N+1, dtype=float)
vel = np.zeros(N+1, dtype=float)

pos = Cond_inic(pos, 0, L, np.sin)
# vel = cond_inic(vel, 0, L, menos_seno)

pos, vel = Evolucao(t, pos, vel, N, dx, dt)

Plot2d(pos, L, t*dt, limites=lim)
