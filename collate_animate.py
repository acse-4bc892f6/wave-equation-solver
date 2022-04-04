import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from IPython import display

# load domain data in order to load grid data
domain_data = np.loadtxt("./output/domain.txt", dtype=float)
# dt_out is the timestep when grid data is written to file
# t_max is the maximum time step
# dt is time step
dt_out, t_max, dt = domain_data[6], domain_data[7], domain_data[8]
# number of processes used to run the program
p = int(domain_data[5])
# number of points in i and j direction in domain
imax, jmax = int(domain_data[0]), int (domain_data[1])
# dimension of domain
ymax, xmax = int(domain_data[2]), int(domain_data[3])

# create list of times to loop over file names
t = 0
t_out = 0
times = []

# adapted from https://www.kite.com/python/answers/how-to-print-a-float-with-two-decimal-places-in-python

while t < t_max:
    if t_out <= t:
        # format in which grid files are written
        times.append("{:.2f}".format(t))
        t_out += dt_out
    t += dt

# load the start and end point of each decomposed block 
parameters_data = np.loadtxt("./output/parameters.txt", dtype=int)

# allocate data to store all grids
all_grids = np.ndarray(shape=(len(times), imax, jmax))

# read in grid data from each file and store in numpy ndarray
for time in range (len(times)):
    collated_grid = np.zeros((imax,jmax))
    for id in range(p):
        fname = "./output/output_" + str(id) + "_" + str(times[time]) + ".txt"
        grid = np.loadtxt(fname, dtype=float)
        i_start, i_end = parameters_data[id][0], parameters_data[id][1]
        j_start, j_end = parameters_data[id][2], parameters_data[id][3]
        collated_grid[i_start:i_end, j_start:j_end] = grid
    all_grids[time, :, :] = collated_grid

# make animation

# adapted from https://www.tutorialspoint.com/how-to-animate-a-seaborn-heatmap-or-correlation-matrix-matplotlib#:~:text=Make%20a%20dimension%20tuple.,dataset%20and%20create%20a%20heatmap.

plt.rcParams["figure.figsize"] = [7, 5]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()
plt.imshow(all_grids[0, :, :], cmap='viridis', extent = [0, xmax, ymax, 0])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')

def init():
    plt.imshow(all_grids[0, :, :], cmap='viridis', extent = [0, xmax, ymax, 0])

def animate(i):
    plt.imshow(all_grids[i, :, :], cmap='viridis', extent = [0, xmax, ymax, 0])

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(times), repeat=False)

# adapted from https://holypython.com/how-to-save-matplotlib-animations-the-ultimate-guide/
writergif = animation.PillowWriter(fps=30) 
anim.save("animation.gif" , writer=writergif)

plt.close()

