from Onda2d_lib import *

N = 100
t = 1
L = 1
dx = L/N
dt = 0.1*(dx*dx)
lim = [-1, 1]

pos = np.zeros((N+1, N+1), dtype=float)
vel = np.zeros((N+1, N+1), dtype=float)

# pos = Cond_cont2d(pos, valorE=1)

# pos = Cond_inic2d_focada(pos, lado=[100, 100], posi=[0, 0], soma=1)

pos = Cond_inic2d_circular(pos, 10, [int(N/2), int(N/2)], tamanho=1)


pos, vel = Evolucao2d_neumann(t, pos, vel, N, dx, dt, constante=1, coord_polares=0)

Plot3d(pos, L, t*dt, limites=lim, cor=0)
