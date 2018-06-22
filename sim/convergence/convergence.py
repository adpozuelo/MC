# adpozuelo@uoc.edu

import sys
import re
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import csv

def readData(filename):
    with open(filename) as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ')
        x = list()
        y = list()
        next(myreader)
        next(myreader)
        for row in myreader:
            if len(row) > 1:
                for item in row:
                    if item:
                        x.append(float(item))
            y.append(x);
            x = []
    return np.array(y)

y1 = readData('./serial/NPT_lat/results/thermoins.dat')
y2 = readData('./serial/NPT_dlp/results/thermoins.dat')

x = range(len(y1[:,3]))
plt.plot(x, y1[:,3], label="lattice")
plt.plot(x, y2[:,3], label="dlp")
plt.title('NPT - SERIAL')
plt.xlabel('Steps')
plt.ylabel('Energy (eV)')
plt.legend()
plt.show()

y1 = readData('./gpu/NPT_lat/results/thermoins.dat')
y2 = readData('./gpu/NPT_dlp/results/thermoins.dat')

x = range(len(y1[:,3]))
plt.plot(x, y1[:,3], label="lattice")
plt.plot(x, y2[:,3], label="dlp")
plt.title('NPT - GPU')
plt.xlabel('Steps')
plt.ylabel('Energy (eV)')
plt.legend()
plt.show()

y1 = readData('./serial/NVT_lat/results/thermoins.dat')
y2 = readData('./serial/NVT_dlp/results/thermoins.dat')

x = range(len(y1[:,2]))
plt.plot(x, y1[:,2], label="lattice")
plt.plot(x, y2[:,2], label="dlp")
plt.title('NVT - SERIAL')
plt.xlabel('Steps')
plt.ylabel('Energy (eV)')
plt.legend()
plt.show()
y1 = readData('./gpu/NVT_lat/results/thermoins.dat')
y2 = readData('./gpu/NVT_dlp/results/thermoins.dat')

x = range(len(y1[:,2]))
plt.plot(x, y1[:,2], label="lattice")
plt.plot(x, y2[:,2], label="dlp")
plt.title('NVT - GPU')
plt.xlabel('Steps')
plt.ylabel('Energy (eV)')
plt.legend()
plt.show()
