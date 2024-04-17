import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit 
from sigfig import round
from sklearn.metrics import r2_score
plt.rcParams.update({'font.size': 28, 'lines.linewidth': 3, 
                    'figure.figsize':[16.0, 9.0], 'figure.dpi':75,
                    'lines.markersize':15})

gaussienne = lambda x, A, mu, sigma,b: b+A*np.exp (-((x-mu)/sigma)**2/2)
fctlineaire = lambda x, a, b: a*x+b

def identification_pic(file,anglemin,anglemax,nbr_pts,nbr_pics=5):
    '''
    Fonction permettant d'identifier les pics d'un spectre et de les ajuster par une gaussienne
    file : str, nom du fichier à analyser
    anglemin : float, valeur minimale de l'angle en degrés
    anglemax : float, valeur maximale de l'angle en degrés
    nbr_pts : int, nombre d'acquisitions
    nbr_pics : int, nombre de pics à identifier
    return :
    anglepic : np.array, angles des pics identifiés
    errpic : np.array, incertitudes sur les angles des pics identifiés
    Lambda : np.array, longueurs d'onde des pics identifiés
    '''
    #lecture du fichier
    angle = np.linspace(anglemin,anglemax,nbr_pts)
    a=np.loadtxt(file, skiprows=18, max_rows=nbr_pts )
    #création des arrays vides
    anglepic= np.zeros(nbr_pics)
    errpic = np.zeros(nbr_pics)
    if file == "nacl.xry":
        #assignation des longueurs d'onde des pics connus par la théorie
        Lambda = [71.08, 126.12, 142.16, 198.18, 213.24 ]
        #identification des pics, de la largeur et de la hauteur des pics
        peaks, properties = find_peaks(a,width=1, prominence=25)
    if file =="lif.xry":
        Lambda = [63.06,71.08,126.12, 142.16, 198.18, 213.24]
        peaks, properties = find_peaks(a,width=1, prominence=10)

    #arrodnissement des indices des pics
    properties["left_ips"],properties["right_ips"]=(np.rint(properties["left_ips"])).astype(int),(np.rint(properties["right_ips"])).astype(int)
    plt.plot(angle, a, label="données expérimentales")
    for i in range(len(peaks)):
        #création des différents paramètres des pics pour l'ajustement gaussien
        #index de la valeurcentrale du pic
        centre_idx= int((properties["left_ips"][i]+properties["right_ips"][i]+1)/2)
        #index de la valeur gauche et droite du pic
        gauche_idx = int(properties["left_ips"][i]-(centre_idx-properties["left_ips"][i]))
        droite_idx = int(properties["right_ips"][i]+(properties["right_ips"][i]-centre_idx))
        #intervalle de la liste d'intensité contenant les valeurs du pic
        intervalle = a[gauche_idx:droite_idx+1]
        #largeur du pic
        largeur = angle[gauche_idx]-angle[droite_idx]
        #intervalle de l'angle du pic
        x= angle[gauche_idx:droite_idx+1]
        #estimation des paramètres initiaux de l'ajustement gaussien
        A_guess = np.max(intervalle)
        mu_guess = np.median(x)
        sigma_guess = np.std(x)
        b_guess = np.min(intervalle)

    # Utilisation de ces estimations initiales
        p0_guess = [A_guess, mu_guess, sigma_guess, b_guess]
        #définition des bornes pour l'ajustement gaussien
        bounds: tuple[list[float]] = ([0, angle[gauche_idx], 1,30], [2*peaks[i], angle[droite_idx], largeur,500])
        #ajustement gaussien
        popt, pcov = curve_fit (gaussienne,x , intervalle, p0=p0_guess, sigma=np.sqrt(intervalle),maxfev=10000)
        #récupération des paramètres de l'ajustement pour créer la gaussienne
        anglefit = np.linspace(angle[gauche_idx],angle[droite_idx+1],100)
        afit=gaussienne(anglefit,*popt)
        #récupération de l'incertitude de l'angle du pic
        err = np.sqrt(np.diag(pcov))
        #idetification du pic de la gaussienne créée
        peakfit,_= find_peaks(afit)
        #récupération de l'angle du pic et de l'incertitude
        anglepic[i]=anglefit[peakfit]
        errpic[i]=err[1]
        #affichage des ajustements gaussiens
        if i== 0:
            plt.plot(anglefit, afit, linestyle="--", color="red",label="fit")
        else:
            plt.plot(anglefit, afit, linestyle="--", color="red")
        plt.scatter(anglefit[peakfit], afit[peakfit],color="k")
        plt.errorbar(anglefit[peakfit],afit[peakfit],xerr=err[1])
    plt.xlabel("Angle (°)")
    plt.ylabel("Intensité (comptes moyens/s)")
    plt.legend()
    plt.show()
    return anglepic, errpic,Lambda

def main():
    #création d'un array représentant le sinus de l'angle
    x= np.linspace(0,0.6,1000)
    #acquisition des données des angles des pcis et de leurs longueurs d'onde pour NaCl
    anglenacl, errnacl,Lambdanacl = identification_pic("nacl.xry", 4, 24, 200)
    #calcul du sinus de l'angle et de l'incertitude
    anglenacl= np.sin(np.radians(anglenacl))
    errnacl = np.sin(errnacl)
    #fit linéaire des données
    poptnacl, pcovnacl = curve_fit(fctlineaire,anglenacl, Lambdanacl )
    errpcovnacl = np.sqrt(np.diag(pcovnacl))
    naclfit= fctlineaire(x, *poptnacl)
    #vérification du coefficient de détermination du fit par rapport aux données
    verifnacl= r2_score(Lambdanacl, fctlineaire(anglenacl,*poptnacl))

#acquisition des données des angles des pcis et de leurs longueurs d'onde pour LiF
    anglelif, errlif,Lambdalif = identification_pic("lif.xry", 3, 33, 300,nbr_pics=6)
    #calcul du sinus de l'angle et de l'incertitude
    anglelif= np.sin(np.radians(anglelif))
    errlif = np.sin(errlif)
    #fit linéaire des données
    poptlif, pcovlif = curve_fit(fctlineaire,anglelif, Lambdalif )
    errpcovlif = np.sqrt(np.diag(pcovlif))
    liffit= fctlineaire(x, *poptlif)
    #vérification du coefficient de détermination du fit par rapport aux données
    veriflif= r2_score(Lambdalif, fctlineaire(anglelif,*poptlif))


    plt.scatter(anglenacl,Lambdanacl, label="NaCl",marker="o",color="tab:orange")
    plt.errorbar(anglenacl,Lambdanacl,xerr=errnacl,linestyle='none',capsize=5, capthick= 3, color="tab:orange")
    plt.plot(x, naclfit,label=f"NaCl: ({round(poptnacl[0],errpcovnacl[0],decimals=1)})x-({round(abs(poptnacl[1]),errpcovnacl[1],decimals=1)}),$R^2$={round(verifnacl,3)}", color="tab:orange")
    print(poptnacl,errpcovnacl)
    plt.scatter(anglelif,Lambdalif, label="LiF",marker="^",color="tab:blue")
    plt.errorbar(anglelif,Lambdalif,xerr=errlif,linestyle='none',capsize=3, color="tab:blue")
    plt.plot(x, liffit,label=f"LiF: ({round(poptlif[0],uncertainty=errpcovlif[0])})x+({round(poptlif[1],uncertainty=errpcovlif[1])}),$R^2$={round(veriflif,3)}", color="tab:blue", linestyle="--")
    print(poptlif,errpcovlif)
    plt.xlabel("sin(θ)")
    plt.ylabel("n$\lambda$(pm)")
    plt.legend()
    plt.show()
    print(poptnacl[0],errpcovnacl[0],poptlif[0],errpcovlif[0])


if __name__ == "__main__":
    main()




