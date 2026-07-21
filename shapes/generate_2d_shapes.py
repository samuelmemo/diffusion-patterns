import torch 
import matplotlib.pyplot as plt
import numpy as np 


def generate_line(m, b, x_range, step_size): 
    point_count = (x_range // step_size) + 1
    points = np.empty([point_count,2])

    i=0
    for x in range(0, x_range+1, step_size):
        y = m*x +b
        points[i,:] = [x,y]
        i+= 1
    return points

def generate_spiral(a,b,lenght): 
    #the "lenght" parameter increases the size of the spiral
    spiral_coords = []
    for theta in np.arange(0,lenght*np.pi,0.1):
        r = a*np.exp(b*theta)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        spiral_coords.append([x,y])
    return np.array(spiral_coords)

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
    #Generate the golden spiral 
    phi = (1 + np.sqrt(5)) /2
    a = 1e-42
    b = 2 * np.log(phi) / np.pi
    sp_points = generate_spiral(a,b,100)
    print_figure(sp_points)


if __name__ == "__main__": 
    main()