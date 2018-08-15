import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def plot_():
    path = "../gcca_data/weight/"
    marker = ["x", "v", "o", "."]
    for i in range(3):
        for j, name in enumerate(os.listdir(path)):
            with open(path + name, "rb") as f:
                u = pickle.load(f)
            u = u[i].reshape(-1)
            u_sort = np.sort(u)
            x = np.arange(len(u))
            greater = np.abs(u_sort) > 1e-5
            lower = np.abs(u_sort) < 1e-5
            # print(u_sort[greater])
            plt.plot(x[greater], u_sort[greater], marker[2],label="greater")
            plt.plot(x[lower], u_sort[lower], marker[2], label="lower")

            filename = name.split(".")[0] + ":W"+str(i)
            plt.title(filename)
            plt.legend(numpoints=1)
            # plt.show()
            plt.savefig("../gcca_data/image/"+filename + ".jpg")
            plt.clf()



plot_()

