from os import system
import cv2
from Onda_lib import *

N = 100
L = 1
dx = L/N
dt = 0.1*(dx*dx)
lim = [-1.01, 1.01]

pos = np.zeros(N+1, dtype=float)
vel = np.zeros(N+1, dtype=float)
# pos = Cond_inic(pos, 0, L, np.sin)
for i in range(25, int(N/2)):
    pos[i] = Reta(i*dx, 1, -0.25)
for i in range(int(N/2), 75):
    pos[i] = Reta(i*dx, -1, 0.75)

fps = 10.0
total_passos = 200000
tempo_video = 15
dimensao = (640, 480)
tamanho_barra = L

x = tempo_video/(1/fps)
passo_por_frame = round((total_passos)/(x))
video = cv2.VideoWriter('animacao1d.avi', cv2.VideoWriter_fourcc(*'mp4v'), fps, dimensao)

for t in range(0, round(x)):
    tempo = t*dt
    pos, vel = Evolucao(passo_por_frame, pos, vel, N, dx, dt)
    Save(pos, t*passo_por_frame*dt, L, 'grafico{}.jpg'.format(''), 'JPG', limites=lim)
    frame = cv2.imread('grafico{}.jpg'.format(''))
    frame = cv2.resize(frame, dsize=dimensao, interpolation=cv2.INTER_CUBIC)
    video.write(frame)
    system("cls")
    print("Fazendo video...")
    print(str(100*t/x) + "%")
print("Video concluido!")

video.release()
cv2.destroyAllWindows()
