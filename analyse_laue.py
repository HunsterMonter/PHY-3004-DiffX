"""
@author: Justin Hamel

https://github.com/juham58/PHY-3003---Diffraction-rayons-X
"""

from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re

plt.rcParams.update({'font.size': 16})

cristaux = {"NaCl": 564.02, "LiF": 402.80, "Si": 543.10}

def laue_import(distance_cristal: float, fichier: bytes, cristal: str) -> pd.DataFrame:
    L = distance_cristal+8
    sigma_L = 2
    a = cristaux[cristal]

    data = pd.read_csv(os.fsdecode(fichier), index_col=0)

    data["Z"] = (np.sqrt(data["X"]**2 + data["Y"]**2 + L**2) - L + 1e-308)
    r = np.sqrt(data["X"]**2 + data["Y"]**2)

    data["u"] = data["X"] / data["Z"]
    data["v"] = data["Y"] / data["Z"]

    data["sigma_u"] = np.sqrt((data["X"]**2 - data["Z"]*(data["Z"]+L))**2 * data["sigma_X"]**2 + (data["X"]*data["Y"])**2 * data["sigma_Y"]**2 + (data["X"]*data["Z"])**2 * sigma_L**2) / (data["Z"]**2 * (data["Z"] + L))
    data["sigma_v"] = np.sqrt((data["X"]*data["Y"]*data["sigma_X"])**2 + (data["Y"]**2-data["Z"]*(data["Z"]+L))**2*data["sigma_Y"]**2 + (data["Y"]*data["Z"]*sigma_L)**2) / (data["Z"]**2 * (data["Z"] + L))

    data["h"] = np.rint(data["u"]).astype("int64")
    data["k"] = np.rint(data["v"]).astype("int64")
    data["l"] = 1

    data[["h", "k", "l"]] = data[["h", "k", "l"]].where((data["h"] % 2 == 1) & (data["k"] % 2 == 1), 2*data[["h", "k", "l"]])

    
    data["n"] = data["h"]**2 + data["k"]**2 + data["l"]**2
    data["d_hkl"] = a / np.sqrt(data["h"]**2 + data["k"]**2 + data["l"]**2)

    data["theta"] = 0.5 * np.arctan(r/L)
    data["sigma_theta"] = np.sqrt((L*data["X"]*data["sigma_X"])**2 + (L*data["Y"]*data["sigma_Y"])**2 + (r**2*sigma_L)**2) / (2*r*(r**2+L**2))

    data["lambda_the"] = 2 * data["d_hkl"] * np.sin(np.arctan (data["l"] / np.sqrt (data["h"]**2 + data["k"]**2)))
    data["lambda_exp"] = 2 * data["d_hkl"] * np.sin(0.5 * np.arctan(np.sqrt(data["X"]**2 + data["Y"]**2) / L))
    data["sigma_lambda"] = np.sqrt(4*data["d_hkl"]**2-data["lambda_exp"]**2) * data["sigma_theta"]
    data["lambda_error"] = np.abs((data["lambda_exp"] - data["lambda_the"]) / data["lambda_the"]) * 100

    minmax_csv = os.fsencode("Resultats/minmax.csv")
    minmax = pd.read_csv(os.fsdecode(minmax_csv), index_col=0)
    nom = os.fsdecode(os.path.basename(fichier)).removesuffix(".csv")
    i = minmax.index[minmax["Nom"] == nom].tolist()

    theta_max = np.max(data["theta"])
    i_theta_max = data.index[data["theta"] == theta_max].tolist()[0]
    sigma_theta_max = data["sigma_theta"].loc[i_theta_max]

    lambda_min = np.min(data["lambda_the"])
    i_lambda_min = data.index[data["lambda_the"] == lambda_min].tolist()[0]
    sigma_lambda_min = data["sigma_lambda"].loc[i_lambda_min]

    if i:
        index = i[0]
        minmax.loc[index] = [nom, 180*theta_max/np.pi, 180*sigma_theta_max/np.pi, lambda_min, sigma_lambda_min]
    else:
        temp = pd.DataFrame({"Nom": nom, "theta_max": 180*theta_max/np.pi, "sigma_theta_max": 180*sigma_theta_max/np.pi, "lambda_min": lambda_min, "sigma_lambda_min": sigma_lambda_min}, index=[len(minmax)])
        minmax = pd.concat([minmax, temp], ignore_index=True)

    minmax.to_csv(os.fsdecode(minmax_csv))

    return data

def laue_mass_import(cristal: str):
    data_dir = os.fsencode("Data/" + cristal)
    resultats_dir = os.fsencode("Resultats/" + cristal)

    for fichier in os.listdir(data_dir):
        if os.fsdecode(fichier).endswith(".csv"):
            data = os.path.join(data_dir, fichier)
            resultats = os.path.join(resultats_dir, fichier)

            distance_str = os.fsdecode(fichier).split("_")[1]
            distance = float(re.findall("\d+", distance_str)[0])

            laue_import(distance, data, cristal).to_csv(os.fsdecode(resultats))


def laue_graph(cristal: str):
    resultats_dir = os.fsencode("Resultats/" + cristal)
    images_dir = os.path.join(resultats_dir, os.fsencode("Images"))

    mean_dataframe = pd.DataFrame()
    index = 0

    for fichier in os.listdir(resultats_dir):
        if os.fsdecode(fichier).endswith(".csv"):

            resultats = os.path.join(resultats_dir, fichier)
            image_name = os.fsencode(os.fsdecode(fichier).removesuffix(".csv") + ".pdf")
            image = os.path.join(images_dir, image_name)

            index += 1
            data = pd.read_csv(os.fsdecode(resultats), index_col=0)
            mean_values = pd.DataFrame({"Nom": os.fsdecode(fichier), "Moyenne": data["lambda_error"].mean(), 
                                        "Écart type": data["lambda_error"].std()}, index = [index])
            mean_dataframe = pd.concat([mean_dataframe, mean_values], ignore_index=True)

            distance_str = os.fsdecode(fichier).split("_")[1]
            distance = re.findall("\d+", distance_str)[0]

            voltage_str = os.fsdecode(fichier).split("_")[2]
            voltage = re.findall("\d+", voltage_str)[0]

            nb_images_str = os.fsdecode(fichier).split("_")[3]
            nb_images = re.findall("\d+", nb_images_str)[0]

            table_title = f"Données acquises avec le cristal de {cristal}, à une distance de {distance} mm, \nune tension au pic de {voltage} kV et un nombre d'images moyennées de {nb_images}."

            # Imprime les données brutes en LaTeX
            #print(data.to_latex(columns=["X", "Y", "Z", "u", "v", "h", "k", "l", "n", "d_hkl", "lambda_exp", "lambda_the", "lambda_error"], caption=table_title, label="tab:" + os.fsdecode(fichier) column_format="ccccccccccccc"))
            
            plt.figure(figsize=(7,7))
            plt.style.use("ggplot")
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
            plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
            #plt.scatter(data["u"], data["v"])
            plt.errorbar(data["u"], data["v"], xerr=data["sigma_u"], yerr=data["sigma_v"], fmt="o")
            plt.xlabel(r"$u=h/l$")
            plt.ylabel(r"$v=k/l$")
            #plt.title(table_title)
            plt.savefig(os.fsdecode(image))
            plt.close()

            # Imprime le tableau de l'annexe B
            #print(mean_dataframe.to_latex(label="tab:tableau_erreurs_moyennes", column_format="|l|r|r|r|"))


def main():
    # Importe les données en format DataFrame
    laue_mass_import("LiF")
    laue_mass_import("NaCl")
    laue_mass_import("Si")

    # Affiche les graphiques et les tableaux
    laue_graph("NaCl")
    laue_graph("LiF")
    laue_graph("Si")

    #plt.show()


if __name__ == "__main__":
    main()
