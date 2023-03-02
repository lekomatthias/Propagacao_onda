from os import system
import cv2
from Onda2d_lib import *

# pyinstaller --onefile nome.py
# -w permite GUI

N = int(input('digite o numero de divisoes: '))
L = float(input('digite o tamanho do lado: '))
dx = L/N
dt = 0.1*(dx*dx)
pos = np.zeros((N+1, N+1), dtype=float)
vel = np.zeros((N+1, N+1), dtype=float)

polares = int(input('coordenadas cartesianas[0] coordenadas polares[1] '))
if polares == 0:
    lados = np.zeros(4, dtype=int)
    lados[0] = float(input('valor do lado esquerdo: '))
    lados[1] = float(input('valor do lado direito: '))
    lados[2] = float(input('valor do lado cima: '))
    lados[3] = float(input('valor do lado baixo: '))
    Cond_cont2d(pos, valorE=lados[0], valorD=lados[1], valorC=lados[2], valorB=lados[3])
neumann = int(input('borda fixa[0] borda variada[1] '))

lim = np.zeros(2, dtype=float)
lim[0] = float(input('digite o limite inferior do grafico: '))
lim[1] = float(input('digite o limite superior do grafico: '))

while(True):
    chave = int(input('deseja colocar uma perturbacao em MONTANHA extra?\n[0 - nao] [1 - sim] '))
    if chave == 0:
        break
    posx_perturb = int(input('digite a posicao inicial da perturbacao em x: '))
    posy_perturb = int(input('digite a posicao inicial da perturbacao em y: '))
    tam_perturb = int(input('digite o raio da perturbacao inicial: '))
    # pos = Cond_inic2d_focada(pos, lado=[tam_perturb, tam_perturb], posi=[posx_perturb, posy_perturb])
    pod = Cond_inic2d_circular(pos, raio=tam_perturb, posi=[posx_perturb, posy_perturb])

while(True):
    chave = int(input('deseja colocar uma perturbacao em ONDA extra?\n[0 - nao] [1 - sim] '))
    if chave == 0:
        break
    vertical = int(input('horizontal[0] vertical[1] '))
    posy_perturb = int(input('digite a posicao inicial da perturbacao: '))
    tam_perturb = int(input('digite o comprimento da perturbacao inicial: '))
    if vertical == 0:
        pos = Cond_inic2d_focada(pos, lado=[N-1, tam_perturb], posi=[1, posy_perturb], opcx=0, soma=1)
    else:
        pos = Cond_inic2d_focada(pos, lado=[tam_perturb, N-1], posi=[posy_perturb, 1], opcy=0, soma=1)

total_passos = int(input('digite a quantidade de passos a serem dados: '))
fps = int(input('digite a quantidade de quadros por segundo do video: '))
tempo_video = int(input('digite o tempo do video: '))

dimensao = (640, 480)
x = tempo_video/(1/fps)
passo_por_frame = round((total_passos)/(x))
video = cv2.VideoWriter('animacao2d.avi', cv2.VideoWriter_fourcc(*'mp4v'), fps, dimensao)

for t in range(0, round(x)):
    Save2d(pos, L, t*passo_por_frame, dt, 'grafico{}.jpg'.format(''), limites=lim, cor=0)
    if neumann == 0:
        pos, vel = Evolucao2d(passo_por_frame, pos, vel, N, dx, dt, constante=1, coord_polares=polares)
    else:
        pos, vel = Evolucao2d_neumann(passo_por_frame, pos, vel, N, dx, dt, constante=1, coord_polares=polares)
    frame = cv2.imread('grafico{}.jpg'.format(''))
    frame = cv2.resize(frame, dsize=dimensao, interpolation=cv2.INTER_CUBIC)
    video.write(frame)
    system("cls")
    print("Fazendo video...")
    print(str(100*t/x) + "%")
print("Video concluido!")

video.release()
cv2.destroyAllWindows()
