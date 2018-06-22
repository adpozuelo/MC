# adpozuelo@uoc.edu

import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import csv

def readData(filename, a, b):
    with open(filename) as csvfile:
        myreader = csv.reader(csvfile, delimiter=' ')
        x = list()
        y = list()
        for row in myreader:
            if len(row) > 1:
                #print(row[a],row[b])
                x.append((float(row[a])))
                y.append((float(row[b])))
    return np.array(x), np.array(y)

x1,y1 = readData('./serial/NVT_dlp/results/ehisto.dat', 0, 1)
x2,y2 = readData('./serial/NVT_lat/results/ehisto.dat', 0, 1)

plt.plot(x1,y1,label="DLP")
plt.plot(x2,y2,label="LAT")
plt.title('NVT - SERIAL')
plt.xlabel('Energy (eV)')
plt.ylabel('Percentage')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=2, mode="expand", borderaxespad=0.)
plt.show()

x1,y1 = readData('./gpu/NVT_dlp/results/ehisto.dat', 0, 1)
x2,y2 = readData('./gpu/NVT_lat/results/ehisto.dat', 0, 1)

plt.plot(x1,y1,label="DLP")
plt.plot(x2,y2,label="LAT")
plt.title('NVT - GPU')
plt.xlabel('Energy (eV)')
plt.ylabel('Percentage')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=2, mode="expand", borderaxespad=0.)
plt.show()

x1,y1 = readData('./serial/NPT_dlp/results/rho_histo.dat', 0, 1)
x2,y2 = readData('./serial/NPT_lat/results/rho_histo.dat', 0, 1)

plt.plot(x1,y1,label="DLP")
plt.plot(x2,y2,label="LAT")
plt.title('NPT - SERIAL')
plt.xlabel('Density (npart / A^3)')
plt.ylabel('Percentage')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=2, mode="expand", borderaxespad=0.)
plt.show()

x1,y1 = readData('./serial/NPT_dlp/results/rho_histo.dat', 0, 1)
x2,y2 = readData('./serial/NPT_lat/results/rho_histo.dat', 0, 1)

plt.plot(x1,y1,label="DLP")
plt.plot(x2,y2,label="LAT")
plt.title('NPT - GPU')
plt.xlabel('Density (npart / A^3)')
plt.ylabel('Percentage')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=2, mode="expand", borderaxespad=0.)
plt.show()
