import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True

# matplotlib.rcParams['ps.useafm'] = True
# matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True

def plot_():
    path = "../gcca_data/weight/"
    marker = ["x", "v", "o", "."]
    for j, name in enumerate(os.listdir(path)):
        plt.figure(1, figsize=(10,5))  # change the figsize if you want

        for i in range(3):
            with open(path + name, "rb") as f:
                u = pickle.load(f)
            u = u[i].reshape(-1)
            u_sort =  np.sort(u)
            x = np.arange(len(u))
            greater = np.abs(u_sort) > 1e-5
            lower = np.abs(u_sort) < 1e-5


            plt.subplot(int("13" + str(i+1)))
            plt.plot(x[greater], u_sort[greater], marker[2],markersize=4)  # change the color if you want
            plt.plot(x[lower], u_sort[lower], marker[2],markersize=4)

            filename = " ".join(name.split(".")[0].split("_")) + ":W"+str(i)
            plt.title(filename)
            plt.xlabel("W"+str(i+1))
            # plt.legend(numpoints=1)
        # plt.show()
        filename = name.split(".")[0]
        plt.savefig("../gcca_data/image/"+filename + ".pdf", bbox_inches='tight')  # if you want to save the pdf you should comment `plt.show` and uncomment this line of code
        plt.clf()



plot_()

