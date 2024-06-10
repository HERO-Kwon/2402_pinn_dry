import matplotlib
import matplotlib.pyplot as plt
import numpy as np

TOL = 1e-14


def visualize_heatmap(x, y, heat_list, heat_pre_list, epoch):
    plt.figure(figsize=(18, 25))
    
    for j in range(len(heat_list)):
        num = 3
        for i in range(num):
            plt.subplot(num, 3, i * 3 + 1)
            plt.contourf(x, y, heat_list[j][i], levels=50, cmap=matplotlib.cm.coolwarm)
            plt.colorbar()
            plt.title('True')
            plt.subplot(num, 3, i * 3 + 2)
            plt.contourf(x, y, heat_pre_list[j][i], levels=50, cmap=matplotlib.cm.coolwarm)
            plt.colorbar()
            plt.title('Prediction')
            plt.subplot(num, 3, i * 3 + 3)
            plt.contourf(x, y, heat_pre_list[j][i] - heat_list[j][i], levels=50, cmap=matplotlib.cm.coolwarm)
            plt.colorbar()
            plt.title('Error')
        plt.savefig('figure/epoch' + str(epoch) +  '_' + str(j) + '_' + str(i) + '_pre.png', bbox_inches='tight', pad_inches=0)
        plt.close()
