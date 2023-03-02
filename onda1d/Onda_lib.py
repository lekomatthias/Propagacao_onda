import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import njit

def Menos_seno(x):
    return -np.sin(x)

def Reta(x, inc, base):
    return x*inc + base

def Cond_inic(vetor, valor, L, funct):
    N = len(vetor)
    X = L/(N-1)
    for i in range(1, N):
        vetor[i] = valor + funct(i*X)
    return vetor

# passo de dt no tempo pela eq da onda
@njit
def Passo(pos, vel, N, dx, dt, constante=1):
    aux = pos.copy()

    for i in range (1 ,N):
        # eq da onda
        pos[i] = aux[i] + dt*((dt/(dx*dx))*constante*(aux[i+1]-2*aux[i]+aux[i-1]) + vel[i])
    
    for i in range (1, N):
        vel[i] = (pos[i] - aux[i])/dt

    return pos, vel

# passo de N*dt no tempo pela eq da onda
@njit
def Evolucao(k_final, pos, vel, N, dx, dt, constante=1):
    for _ in range(0, k_final):
        pos, vel = Passo(pos, vel, N, dx, dt, constante=constante)
    return pos, vel

def Plot2d(pos, L, t, limites=[0, 0]):
    abscissas = np.linspace(0, L, len(pos))
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("Posicao \n Tempo = %.3f"%t)
    ax.set_xlabel('$posicao x$', fontsize=12)
    ax.set_ylabel('$posicao y$', fontsize=12)

    if limites != [0, 0]:
        plt.ylim(limites[0], limites[1])

    plt.plot(abscissas, pos, color='red', linewidth=3)
    plt.show()

def Save(pos, t, L, nome, formato, limites=[0, 0]):
    abscissas = np.linspace(0, L, len(pos))
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title('t=%.3f'%t, fontsize=18)
    ax.set_xlabel('$posicao x$', fontsize=18)
    ax.set_ylabel('$posicao y$', fontsize=18)

    if limites != [0, 0]:
        plt.ylim(limites[0], limites[1])

    plt.plot(abscissas, pos, color='red', linewidth=2)
    plt.savefig(nome , format=formato , dpi=200, bbox_inches='tight')
    plt.cla()
    plt.close(fig)
