from functools import partial
from scipy.optimize import curve_fit
import cv2
import imageio
import imagepers
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys


def gaussienne2D(xy, A, mu_x, mu_y, sigma_x, sigma_y):
    x, y = xy
    f = A * np.exp(-((x - mu_x)**2 / sigma_x**2 + (y - mu_y)**2 / sigma_y**2) / 2)
    return f.ravel()


def find_peaks(filename: bytes, per: float, x0: float, y0: float, fig, drop):
    img = imageio.v3.imread(os.fsdecode(filename))

    # Enlève les petites taches blanches parasites
    blurred_img = cv2.medianBlur(img, 5)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(blurred_img, kernel, iterations=1)
    output = cv2.dilate(erosion, kernel, iterations=1)

    # Diminue la résolution et blur pour le calcul de maxima
    scale: int = 5
    res = cv2.resize(output, dsize=(int(1152/scale), int(1300/scale)), interpolation=cv2.INTER_CUBIC)
    blurred_res = cv2.medianBlur(res, 3)

    # Positionne toutes les valeurs entre 0 et 255 dans un array de unint8
    data = np.array(blurred_res, dtype=np.float64)
    rescale = np.array(np.clip(data * 1 * 255 / data.max(), 0, 255), dtype=np.uint8)

    # Détecte les pics
    g0 = imagepers.persistence(rescale)

    peaks = []
    for homclass in g0:
        p_birth, bl, pers, p_death = homclass
        if pers <= per:
            continue
        y, x = p_birth
        peaks.append([scale*x, scale*y])

    peaks = np.array(peaks)
    df = pd.DataFrame(peaks, columns=["x", "y"])

    # Détermine le centre de l'image
    df["r"] = np.sqrt((df["x"] - x0)**2 + (df["y"] - y0)**2)
    df.sort_values("r", inplace=True, ignore_index=True)

    [df.drop(index=i, inplace=True) for i in drop]
    df.reset_index(drop=True, inplace=True)

    if df["r"][0] < 75:
        x_centre = df["x"][0]
        y_centre = df["y"][0]
    else:
        x_centre = x0
        y_centre = y0

    # Enlève les faux positifs sur le halo près du centre
    # et les faux positifs sur les bords de l'image
    df["r"] = np.sqrt((df["x"] - x_centre)**2 + (df["y"] - y_centre)**2)
    df.drop(df[(df["r"] > 0) & (df["r"] < 75)].index, inplace=True)

    df.drop(df[df["x"] < 15].index, inplace=True)
    df.drop(df[df["x"] > 1152-15].index, inplace=True)
    df.drop(df[df["y"] < 15].index, inplace=True)
    df.drop(df[df["y"] > 1300-15].index, inplace=True)

    df.reset_index(drop=True, inplace=True)

    # Masque les pics pour calculer l'écart type du bruit
    img_mask = np.ma.array(img, mask=False)
    img_mask.mask[y_centre-75:y_centre+75, x_centre-75:x_centre+75] = True
    for y, x in zip(df["y"], df["x"]):
        img_mask.mask[y-15:y+15, x-15:x+15] = True

    #plt.figure(layout="constrained")
    #plt.imshow(3*img_mask, cmap="gray", vmin=0, vmax=255)
    #plt.show()

    mu_bruit = img_mask.mean()
    sigma_bruit = img_mask.std()

    # Enlève le masque du centre et le remplace par celui du pic
    img_mask.mask[y_centre-75:y_centre+75, x_centre-75:x_centre+75] = False
    img_mask.mask[y_centre-15:y_centre+15, x_centre-15:x_centre+15] = True
    # Inverse le masque pour obtenir uniquement les pics pour calculer le signal moyen
    img_mask.mask = np.logical_not(img_mask.mask)

    #plt.figure(layout="constrained")
    #plt.imshow(3*img_mask, cmap="gray", vmin=0, vmax=255)
    #plt.show()

    mu_signal = img_mask.mean()

    SNR = mu_signal/sigma_bruit
    CNR = (mu_signal - mu_bruit) / sigma_bruit

    bruit_csv = os.fsencode("Resultats/bruit.csv")
    bruit = pd.read_csv(os.fsdecode(bruit_csv), index_col=0)
    nom = os.fsdecode(os.path.basename(filename)).removesuffix(".tif")
    i = bruit.index[bruit['Nom'] == nom].tolist()

    if i:
        index = i[0]
        bruit.loc[index] = [nom, per, SNR, CNR]
    else:
        temp = pd.DataFrame({"Nom": nom, "Persistence": per, "SNR": SNR, "CNR": CNR}, index=[len(bruit)])
        bruit = pd.concat([bruit, temp], ignore_index=True)

    bruit.to_csv(os.fsdecode(bruit_csv))

    """
    print(f"Signal: mu = {mu_signal}")
    print(f"Bruit:  mu = {mu_bruit}, sigma = {sigma_bruit}")
    print(f"SNR: {SNR}, CNR: {CNR}")
    """

    fits = []
    mu_x = []
    mu_y = []
    sigma_x = []
    sigma_y = []
    # Fit un gaussienne 2D aux points pour calculer la position et l'incertitude
    if not df.empty:
        for x_point, y_point in zip(df["x"], df["y"]):
            x = np.linspace(x_point-15, x_point+14, 30)
            y = np.linspace(y_point-15, y_point+14, 30)
            x, y = np.meshgrid(x, y)

            data = img[y_point-15:y_point+15, x_point-15:x_point+15]
            # +1 car scipy n'est pas capable de fit quand sigma = 0
            sigma = data / SNR + 1

            # Estimés initiaux et bornes des paramètres
            A0 = img[y_point, x_point]
            p0 = (A0, x_point, y_point, 5, 5)
            bounds = ([0, x_point-15, y_point-15, 0, 0], [2*np.max(data), x_point+14, y_point+14, 30, 30])

            popt, pcov = curve_fit(gaussienne2D, (x, y), data.ravel(), p0=p0, bounds=bounds, sigma=sigma.ravel())

            fits.append([x, y, gaussienne2D((y, x), A=popt[0], mu_x=popt[2], mu_y=popt[1], sigma_x=popt[4], sigma_y=popt[3]).reshape(30, 30)])
            mu_x.append(popt[1])
            mu_y.append(popt[2])
            sigma_x.append(popt[3])
            sigma_y.append(popt[4])

    mu_x = np.array(mu_x)
    mu_y = np.array(mu_y)
    sigma_x = np.array(sigma_x)
    sigma_y = np.array(sigma_y)

    # Distances à partir du centre en mm
    if df["r"][0] < 0.1:
        df["X"] = 49.5/1000 * (mu_x - mu_x[0])
        df["Y"] = 49.5/1000 * (mu_y - mu_y[0])
        df["sigma_X"] = 49.5/1000 * np.sqrt(sigma_x**2 + sigma_x[0]**2)
        df["sigma_Y"] = 49.5/1000 * np.sqrt(sigma_y**2 + sigma_y[0]**2)
        df.loc[0, "sigma_X"] = 0
        df.loc[0, "sigma_Y"] = 0
    else:
        df["X"] = 49.5/1000 * (mu_x - x_centre)
        df["Y"] = 49.5/1000 * (mu_y - y_centre)
        df["sigma_X"] = 49.5/1000 * sigma_x**2
        df["sigma_Y"] = 49.5/1000 * sigma_y**2

    X = np.linspace(515, 544, 30)
    Y = np.linspace(615, 644, 30)
    X, Y = np.meshgrid(X, Y)
    f = gaussienne2D((X, Y), 100, 530, 630, 2, 5).reshape(30, 30)

    # Affiche l'image avec les pics détectés
    plt.imshow(3*img, cmap="gray", vmin=0, vmax=255)
    for i, xy in enumerate(zip(df["x"], df["y"])):
        x, y = xy
        plt.plot(x, y, '.', c='b')
        plt.text(x+20, y+10, str(i+1), color='w')
    """
    # Affiche les fits des points
    for x, y, f in fits:
        plt.contour(x, y, f)
    """

    """
    X = np.linspace(0, 260, 1000)

    plt.figure(layout="constrained")
    plt.style.use("ggplot")
    plt.plot(X, X-20)
    plt.plot(X, X, color="grey")
    for i, homclass in enumerate(g0):
        p_birth, bl, pers, p_death = homclass
        if pers <= 1.0:
            continue

        x, y = bl, bl-pers
        plt.plot(x, y, '.', c='b')
        plt.text(x, y+2, str(i+1), color='b')
    plt.xlim(-5,260)
    plt.ylim(-5,260)
    plt.xlabel("Birth level")
    plt.ylabel("Death level")
    plt.show()
    """
    print(df)

    return df, fig


def main():
    # Index à enlever pour les faux positifs
    try:
        index = list(sys.argv[1:])
        index = [int(x)-1 for x in index]
    except:
        index = []

    cristal = "Si"

    Images = os.fsencode(f"Data/{cristal}/Images")
    Figs = os.fsencode(f"Data/{cristal}/Figs")
    Data = os.fsencode(f"Data/{cristal}")

    """
    for file in os.listdir(Images):
        file = os.fsdecode(file)
        if os.fsdecode(file).endswith(".tif"):
            file = file.removesuffix(".tif")
            print(file)

            imgname = os.fsencode(file + ".tif")
            figname = os.fsencode(file + f"_{int(per)}per.pdf")
            csvname = os.fsencode(file + ".csv")

            img = os.path.join(Images, imgname)
            fig = os.path.join(Figs, figname)
            csv = os.path.join(Data, csvname)
            f = plt.figure(layout="constrained")
            df, f = find_peaks(img, per, x0, y0, f)

            # Sauvegarde la figure avec les pics identifiés
            plt.savefig(os.fsdecode(fig))
            # N'affiche pas les images quand on processe toutes les images
            #plt.show()

            df[["X", "Y", "sigma_X", "sigma_Y"]].to_csv(os.fsdecode(csv))
    """
    file = "si_10mm_35kVp_80images"
    per = 20
    #x0 = 576
    #y0 = 650
    x0 = 585
    y0 = 650

    imgname = os.fsencode(file + ".tif")
    figname = os.fsencode(file + f"_{int(per)}per.pdf")
    csvname = os.fsencode(file + ".csv")

    img = os.path.join(Images, imgname)
    fig = os.path.join(Figs, figname)
    csv = os.path.join(Data, csvname)


    f = plt.figure(layout="constrained")
    df, f = find_peaks(img, per, x0, y0, f, index)

    # Sauvegarde et affiche la figure avec les pics identifiés
    # Permet d'ajuster les paramètres pour une image en particulier
    plt.savefig(os.fsdecode(fig))
    plt.show()

    df[["X", "Y", "sigma_X", "sigma_Y"]].to_csv(os.fsdecode(csv))


if __name__ == "__main__":
    main()
