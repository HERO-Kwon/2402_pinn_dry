import matplotlib
import matplotlib.pyplot as plt
import numpy as np

TOL = 1e-14


def visualize_heatmap(x, y, heat_list, heat_pre_list, epoch):
    plt.figure(figsize=(18, 25))
    num = len(heat_list)
    
    for i in range(num):
        plt.subplot(num, 3, i * 3 + 1)
        plt.contourf(x, y, heat_list[i].squeeze(), levels=50, cmap=matplotlib.cm.coolwarm)
        plt.colorbar()
        plt.title('True')
        plt.subplot(num, 3, i * 3 + 2)
        plt.contourf(x, y, heat_pre_list[i].squeeze(), levels=50, cmap=matplotlib.cm.coolwarm)
        plt.colorbar()
        plt.title('Prediction')
        plt.subplot(num, 3, i * 3 + 3)
        plt.contourf(x, y, heat_pre_list[i].squeeze() - heat_list[i].squeeze(), levels=50, cmap=matplotlib.cm.coolwarm)
        plt.colorbar()
        plt.title('Error')
    plt.savefig('figure/epoch' + str(epoch) + '_pre.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    '''
    for i in range(num):
        for j in range(5):  # (5, 64, 128)의 첫 번째 차원을 반복
            plt.figure(figsize=(15, 5))  # 각 반복마다 새로운 figure 생성
            plt.subplot(1, 3, 1)
            plt.contourf(x, y, heat_list[i][j], levels=50, cmap=matplotlib.cm.coolwarm)
            plt.colorbar()
            plt.title(f'True (Slice {j+1})')
            plt.subplot(1, 3, 2)
            plt.contourf(x, y, heat_pre_list[i][j], levels=50, cmap=matplotlib.cm.coolwarm)
            plt.colorbar()
            plt.title(f'Prediction (Slice {j+1})')
            plt.subplot(1, 3, 3)
            plt.contourf(x, y, heat_pre_list[i][j] - heat_list[i][j], levels=50, cmap=matplotlib.cm.coolwarm)
            plt.colorbar()
            plt.title(f'Error (Slice {j+1})')
            plt.show()  # 각 반복마다 figure를 보여줌
    plt.savefig('figure/epoch' + str(epoch) + '_pre.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    '''