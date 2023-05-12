"""
------------------------------------------------------------------------------------
UCL : PROJET P4 RADAR

Groupe 6
Morabito Maxime
Morelle Guillaume
Amerijckx Cédric
Sarge Théophile
Durniez Quentin

------------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sb

#Gestion des outputs
toresults = True
tovideo = False
toprint =  True

# fichier à analyser
filename = ""    # fichier à analyser
O_calib = 90     # angle obtenu sur le fichier de calibrage 90° par défaut
f= np.load(filename + ".npz",allow_pickle = True)
lst = f.files
# récupération de data (informations importante pour distance,vitesse,angle)
try :
    data, data_times, background, background_times, chirp, datetime = f.values()
except:
    data, background, chirp, data_times = f.values()

# Caractéristiques de la prise de données
f0,B,Ns,Nc,Ts,Tc = chirp
nb_antennes = int(data.shape[1]/2)
c = 3e8
frame = 4 # frame utilisée pour les graphiques
dx, dy, dz = 32, 32, 16 # nombre de points pris pour la FFT3
                        # dans les directions distance,vitesse et angle 
xstick, ystick = 50, 50 # pour le graphique
dia = 6.25e-2   # distance inter-antennes d
l = 12.5e-2     # longueur d'onde utilisée
if(toprint) :
    for item in lst:
        print(item)
Ms = int(Ns)            # nombre de points par chirp
Mc = int(Nc)            # nombre de chirp par frame
Mp = int(Tc/Ts - Ms)    # nombre de points de pause entre deux chirps
Nframe = len(data)
N = len(data[0][0])

if(toprint):
    print("Ms : {0}".format(Ms))
    print("Mc : {0}".format(Ms))
    print("Mp : {0}".format(Mp))
    print("Nframe : {0}".format(Nframe))
    
# on restructure data
data = data.reshape(Nframe, 4, Mc, Ms + Mp)[:,:4,:,:]
# on enlève l'impact de l'antenne émettrice.
# on retire la moyenne selon les chirps <=> la somme est nulle
# <=> TF(chirp à f = 0) = 0
# <=> couper la composante à vitesse nulle.
data = data-np.mean(data, axis=2).reshape(Nframe, 4, 1, Ms+Mp) 

# on retire le background, semble inutile (devrait être redondant avec l'étape précédente)
#data = data - np.mean(background,axis = 0).reshape(1,4, Mc, Ms + Mp)

if(toprint):
    print("data.shape avec pauses "+ str(data.shape))
# on retire les pauses entre les chirps
data = data[:,:,:,:Ms]

if(toprint):
    print("data.shape sans pauses "+ str(data.shape))
    
def enveloppe(I, Q):
    return I-1j*Q

# on remplace les données I et Q par l'enveloppe I-jQ pour chaque antenne
enveloppe = np.zeros((Nframe, 2, data.shape[2], data.shape[3])) - 1j * np.zeros((Nframe, 2, data.shape[2], data.shape[3]))
enveloppe[:,0,:,:] = data[:,0,:,:] - 1j*data[:,1,:,:]
enveloppe[:,1,:,:] = data[:,2,:,:] - 1j*data[:,3,:,:]

# Correction de l'enveloppe de l'antenne 2 (Calibrage de l'angle)
    # facteur correctif
W_calib = np.exp(-1j * 2*np.pi * np.cos(O_calib/180*np.pi) * dia/l )
    #correction
enveloppe[:,1,:,:] = enveloppe[:,1,:,:]*W_calib

# On effectue la FFT3
# FFT(sur dx points,selon Ms => axe 3) contribution de la distance
# FFT(sur dy points,selon Ms => axe 2) contribution de la vitesse
# FFT(sur dz points,selon Ms => axe 1) contribution de l'angle
data_fft = np.fft.fftn(enveloppe[:,:,:,:], s=(dx,dy,dz) , axes = (3,2,1))

# retrait de la composante à vitesse nulle, semble ne pas marcher
#(devrait être redondant le retrait fait à priori)
#data_fft[:][0][:] = 0+0j

#conversion fréquence -> distance
d = np.fft.fftshift(np.fft.fftfreq(dx, (2*B)/(c*Ns)))
d-=d[0] # on fixe la référence de distance à 0
d = np.round(d,2)
#conversion fréquence -> vitesse
v = np.round(np.fft.fftshift(np.fft.fftfreq(dy, (2*f0*Tc)/c)), 2)  
#conversion fréquence -> angle
o = np.round(np.arccos(np.fft.fftshift(np.fft.fftfreq(dz,-dia/l)))/np.pi*180)
# on shift les fréquences pour les centrées autour de 0 (angle et vitesse),
# il faut shifter les données aussi.
data_fft = np.fft.fftshift(data_fft, axes=(1,2))
# on regarde la norme de la FFT
data_norm = np.abs(data_fft)**2 

if(toresults):
    # pour quelques frames on regarde les résultats
    O_moy = 0
    i_o_max, i_v_max, i_d_max = 0,0,0;
    C_max = 0
    O_array = []
    for i in range(Nframe):
        I_frame = i
        # récupération des index du maximum
        Cf_max = np.max(data_norm[I_frame])
        ind = np.unravel_index(np.argmax(data_norm[I_frame]),data_norm[0].shape)
        if(Cf_max > C_max): 
            C_max = Cf_max
            i_o_max, i_v_max, i_d_max = ind       
        print("frame : "+ str(I_frame)+ " "+
              "ditsance : "+ str(d[ind[2]]) +" [m], "+
              "vitesse : " +str(v[ind[1]]) +" [m/s], "+
              "angle : " +str(o[ind[0]]) +" [°]")
        
        O_array.append([o[ind[0]]])
    O_moy = np.round(O_moy/Nframe,2)
    print("angle moyen : "+str(O_moy) +" [°]")
    print('Corrélation maximale : '+str(C_max))
# Valeur moyenne de l'angle avec le fichier Calib0:
#O_calib = 122.9 # [°]

# Graphique  distance - vitesse
fig = plt.figure()
xticksStep = xstick
yticksStep = ystick
Image = plt.imshow(data_norm[frame,i_o_max], extent = (d[0], d[-1], v[0], v[-1]))
plt.colorbar()
plt.title("Corrélation")
plt.xlabel("Distance [m]")
plt.ylabel("Vitesse [m/s]")
plt.savefig(filename + ".pdf")
plt.show()

# Graphique distance angle
fig = plt.figure()
xticksStep = xstick
yticksStep = ystick
Image = plt.imshow(data_norm[frame,:,i_v_max,:], extent = (o[0],o[-1],d[0], d[-1]))
plt.colorbar()
plt.title("Corrélation")
plt.xlabel("angle [°]")
plt.ylabel(" distance [m]")
plt.savefig(filename + ".pdf")
plt.show()

# video (frame by frame)
if(tovideo):
    def video(i):
        Image.update({"data":data_norm[i, 0]})
    
    vid = FuncAnimation(fig, video, frames=Nframe, repeat=False)
    try:
        vid.save(filename + ".mp4", fps = 10)
    except:
        vid.save(filename + ".gif", fps = 10)



