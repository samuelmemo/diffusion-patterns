import torch 
import torchvision 
import matplotlib.pyplot as plt
import numpy as np 


def generate_line(x_range, step_size): 
    m = 2
    b = 1
    point_count = (x_range // step_size) + 1
    points = np.empty([point_count,2])

    i=0

    for x in range(0, x_range+1, step_size):
        y = m*x +b
        points[i,:] = [x,y]
        i+= 1
    return points

def print_figure(points): 
    fig, ax = plt.subplots() 
    x = points[:,0]
    y = points[:,1]
    ax.plot(x,y, 'o')
    ax.set_title("Simple Waveform Visualization", fontsize=14)

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    #ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def main():
    points = generate_line(200, 10)
    print(points)
    print_figure(points)


if __name__ == "__main__": 
    main()