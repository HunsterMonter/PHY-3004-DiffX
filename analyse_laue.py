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
    L = distance_cristal
    a = cristaux[cristal]

    data = pd.read_csv(os.fsdecode(fichier), index_col=0)

    #data["X"] = (data["X"] - data.loc[1, "X"])
    #data["Y"] = (data["Y"] - data.loc[1, "Y"])
    data["Z"] = (np.sqrt(data["X"]**2 + data["Y"]**2 + L**2) - L + 1e-308)

    data["u"] = data["X"] / data["Z"]
    data["v"] = data["Y"] / data["Z"]

    data["h"] = np.rint(data["u"]).astype("int64")
    data["k"] = np.rint(data["v"]).astype("int64")
    data["l"] = 1

    data[["h", "k", "l"]] = data[["h", "k", "l"]].where((data["h"] % 2 == 1) & (data["k"] % 2 == 1), 2*data[["h", "k", "l"]])

    
    data["n"] = data["h"]**2 + data["k"]**2 + data["l"]**2
    data["d_hkl"] = a / np.sqrt(data["h"]**2 + data["k"]**2 + data["l"]**2)

    data["lambda_exp"] = 2 * data["d_hkl"] * np.sin(0.5 * np.arctan(np.sqrt(data["X"]**2 + data["Y"]**2) / L))
    data["lambda_the"] = 2 * data["d_hkl"] * np.sin(np.arctan (data["l"] / np.sqrt (data["h"]**2 + data["k"]**2)))
    data["lambda_error"] = np.abs((data["lambda_exp"] - data["lambda_the"]) / data["lambda_the"]) * 100

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
            plt.scatter(data["u"], data["v"])
            plt.xlabel(r"$u=h/l$")
            plt.ylabel(r"$v=k/l$")
            plt.title(table_title)
            plt.savefig(os.fsdecode(image))

            # Imprime le tableau de l'annexe B
            #print(mean_dataframe.to_latex(label="tab:tableau_erreurs_moyennes", column_format="|l|r|r|r|"))


def main():
    # Importe les données en format DataFrame
    laue_mass_import("LiF")
    laue_mass_import("NaCl")
    laue_mass_import("Si")

    # Affiche les graphiques et les tableaux
    laue_graph("NaCl")

    #plt.show()


if __name__ == "__main__":
    main()
