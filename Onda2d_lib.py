import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import njit

def Const(x, opc=0):
    return opc

def Reta(x, opc=1, base=0):
    return x*opc + base

def Seno(x, opc=1):
    return np.sin(x)*opc

# cria uma matriz personalizada do tamanho desejado
def Matriz_personalizada(lado=[5, 5], largura=np.pi, functx=Const, functy=Const, opcx=1, opcy=1, soma=0):
    matriz = np.zeros((lado[0], lado[1]), dtype=float)
    intervalo = [largura/lado[0], largura/lado[1]]
    for j in range(0, lado[1]):
        for i in range(0, lado[0]):
            if soma == 0:
                matriz[i, j] = functx(i*intervalo[0], opcx) * functy(j*intervalo[1], opcy)
            else:
                matriz[i, j] = functx(i*intervalo[0], opcx) + functy(j*intervalo[1], opcy)
    return matriz

# coloca uma matriz personalizada dentro de uma matriz maior
def Cond_inic2d_focada(matriz, lado=[5, 5], posi=[1, 1], largura=np.pi, functx=Seno, functy=Seno, opcx=1, opcy=1, soma=0):
    matriz[posi[0]:posi[0]+lado[0], posi[1]:posi[1]+lado[1]] += Matriz_personalizada(lado=lado, largura=largura,
                                                                    functx=functx, functy=functy, opcx=opcx, opcy=opcy, soma=soma)
    return matriz

def Matriz_circular(raio, tamanho=1):
    matriz = np.zeros((raio*2, raio*2), dtype=float)
    intervalo = np.pi/(2*raio)
    for j in range(0, raio*2):
        for i in range(0, raio*2):
            if round((i-raio)*(i-raio) + (j-raio)*(j-raio)) < raio*raio:
                hip = np.sqrt((j-raio)*(j-raio) + (i-raio)*(i-raio))
                matriz[i, j] = Seno((raio-hip)*intervalo)

    return matriz

def Cond_inic2d_circular(matriz, raio=10, posi=[5, 5], tamanho=1):
    matriz[posi[0]-raio:posi[0]+raio, posi[1]-raio:posi[1]+raio] += Matriz_circular(raio=raio, tamanho=tamanho)
    return matriz

# muda o contorno da matriz
def Cond_cont2d(vetor2d, valorE=0, valorD=0, valorC=0, valorB=0):
    final = len(vetor2d)
    for i in range(1, final):
        vetor2d[i, 0:2] = valorB
    for i in range(1, final):
        vetor2d[i, final-3:final-1] = valorC
    for i in range(1, final):
        vetor2d[0:2, i] = valorE
    for i in range(1, final):
        vetor2d[final-3:final-1, i] = valorD
    return vetor2d

# passo de dt no tempo pela equacao da onda 2d
@njit
def Passo2d(vetor2d, vel2d, N, dx, dt, constante=1):
    aux = vetor2d.copy()
    for j in range (1, N):
        for i in range (1, N) :
            # influencia horizontal
            h = (dt/(dx*dx))*(aux[i+1, j] - 2*aux[i, j] + aux[i-1, j])
            # influencia vertical
            v = (dt/(dx*dx))*(aux[i, j+1] - 2*aux[i, j] + aux[i, j-1])
            vetor2d[i, j] = aux[i, j] + constante*dt*(h + v + vel2d[i,j])

    for j in range (1, N):
        for i in range (1, N) :
            vel2d[i, j] = (vetor2d[i, j] - aux[i, j])/(dt*constante)

    return vetor2d, vel2d

# passo de dt no tempo pela equacao da onda 2d
# limitada por uma circulo de raio igual a metade do lado da matriz
@njit
def Passo2d_polar(vetor2d, vel2d, N, dx, dt, constante=1):
    aux = vetor2d.copy()
    for j in range (1, N):
        for i in range (1, N) :
            if round((i-N/2)*(i-N/2) + (j-N/2)*(j-N/2)) < round(((N-1)/2)*((N-1)/2)):
                # influencia horizontal
                h = (dt/(dx*dx))*(aux[i+1, j] - 2*aux[i, j] + aux[i-1, j])
                # influencia vertical
                v = (dt/(dx*dx))*(aux[i, j+1] - 2*aux[i, j] + aux[i, j-1])
                vetor2d[i, j] = aux[i, j] + constante*dt*(h + v + vel2d[i,j])

    for j in range (0, N+1):
        for i in range (0, N+1) :
            vel2d[i, j] = (vetor2d[i, j] - aux[i, j])/(dt*constante)

    return vetor2d, vel2d

# passo de N*dt no tempo pela eq da onda 2d
@njit
def Evolucao2d(k_final, matriz, vel2d, N, dx, dt, constante=1, coord_polares=0):
    if coord_polares == 0:
        for _ in range(0, k_final):
            matriz, vel2d = Passo2d(matriz, vel2d, N, dx, dt, constante=constante)
        return matriz, vel2d
    else:
        for _ in range(0, k_final):
            matriz, vel2d = Passo2d_polar(matriz, vel2d, N, dx, dt, constante=constante)
        return matriz, vel2d

# impoe fluxo na borda esqueda
@njit
def Neumann_esquerda(matriz, dx, N, fluxo=0):
    for pos in range (0, N):
        matriz[0, pos] = -2*(fluxo*dx + matriz[2, pos]/2 - 2*matriz[1, pos])/3
    return matriz

# impoe fluxo na borda direita
@njit
def Neumann_direita(matriz, dx, N, fluxo=0):
    for pos in range (0, N):
        matriz[N, pos] = -2*(fluxo*dx + matriz[N-2, pos]/2 - 2*matriz[N-1, pos])/3
    return matriz

# impoe fluxo na borda de cima
@njit        
def Neumann_cima(matriz, dx, N, fluxo=0):
    for pos in range (0, N):
        matriz[pos, N] = -2*(fluxo*dx + matriz[pos, N-2]/2 - 2*matriz[pos, N-1])/3
    return matriz

# impoe fluxo na borda de baixo
@njit
def Neumann_baixo(matriz, dx, N, fluxo=0):
    for pos in range (0, N):
        matriz[pos, 0] = -2*(fluxo*dx + matriz[pos, 2]/2 - 2*matriz[pos, 1])/3
    return matriz

# impoe fluxo na borda do circulo de raio igual a metade do lado da matriz
@njit
def Neumann_polar(matriz, dx, N, fluxo=0):
    for j in range(0, N+1):
        for i in range(0, N+1):
            if round((i-N/2)*(i-N/2) + (j-N/2)*(j-N/2)) >= round(((N-1)/2)*((N-1)/2)):
                x = i-N/2
                y = j-N/2
                hipotenusa = np.sqrt(x*x + y*y)
                sen = y/hipotenusa
                cos = x/hipotenusa
                # manobra feita pra deixar o codigo mais legivel,
                # basicamente foi feito um jogo de variaveis para conseguir atualizar
                # o fluxo nas bordas de na direcao normal a borda.
                posx = [0, round(N/2 + ((N-1)/2 - 1)*cos), round(N/2 + ((N-1)/2 - 2)*cos)]
                posy = [0, round(N/2 + ((N-1)/2 - 1)*sen), round(N/2 + ((N-1)/2 - 2)*sen)]
                matriz[i, j] = -2*(fluxo*dx + matriz[posx[2], posy[2]]/2 - 2*matriz[posx[1], posy[1]])/3
                
    return matriz

# N passos impondo fluxo nas bordas a cada passo
@njit
def Evolucao2d_neumann(k_final, matriz, vel, N, dx, dt, constante=1, coord_polares=0):
    if coord_polares == 0:
        for _ in range (0, k_final):
            # derivadas nas bordas para que o fluxo seja o dado
            Neumann_baixo(matriz, dx, N)
            Neumann_cima(matriz, dx, N)
            Neumann_esquerda(matriz, dx, N)
            Neumann_direita(matriz, dx, N)

            matriz, vel = Passo2d(matriz, vel, N, dx, dt, constante=constante)

    else:
        for _ in range (0, k_final):
            Neumann_polar(matriz, dx, N)

            matriz, vel = Passo2d_polar(matriz, vel, N, dx, dt, constante=constante)

    return matriz, vel

def Plot3d(matriz, L, t, limites=[0, 0], cor=1):
    # criacao do espaco 3d
    X, Y = np.mgrid[0:L:(len(matriz[0]))*1j, 0:L:(len(matriz))*1j]
    try:
        plt.style.use ('default')
    except:
        pass
    ax = plt.axes(projection = '3d')

    if cor == 0:
        ax.plot_surface(X, Y, matriz, cmap = cm.winter)
    elif cor == 1:
        ax.plot_surface(X, Y, matriz, cmap = 'viridis')
    else:
        ax.plot_wireframe(X, Y, matriz, color = 'red')

    ax.set_title(' lado: %.3fm '%L
                +'\ntempo: %.3fs '%t)
    ax.set_ylabel('$Y$')
    ax.set_xlabel('$X$')
    ax.set_zlabel('$Z$')
    if limites != [0, 0]:
        ax.set_zlim(limites[0],limites[1])

    plt.show()

def Save2d(matriz, Largura, tempo, dt, nome, limites=[0, 0], cor=1):
    # criacao do espaco 3d
    X, Y = np.mgrid[0:Largura:(len(matriz[0]))*1j, 0:Largura:(len(matriz))*1j]

    fig = plt.figure() # tentativa de nao criar figura e nao correr o risco de nao deletar
    ax = plt.axes(projection = '3d')
    if cor == 0:
        ax.plot_surface(X, Y, matriz, cmap = cm.winter)
    elif cor == 1:
        ax.plot_surface(X, Y, matriz, cmap = 'viridis')
    else:
        ax.plot_wireframe(X, Y, matriz, color = 'red')
    ax.set_title(' lado: %.3fm '%Largura
                +'\ntempo: %.3fs '%(tempo*dt))
    ax.set_ylabel('$Y$')
    ax.set_xlabel('$X$')
    ax.set_zlabel('$Z$')
    if np.all(limites):
        ax.set_zlim(limites[0],limites[1])

    plt.savefig(nome, dpi=200, bbox_inches='tight')
    plt.cla()
    plt.close(fig) # garantia de que deletei  a figura
