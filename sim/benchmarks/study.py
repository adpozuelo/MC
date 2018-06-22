# adpozuelo@uoc.edu

import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pprint
import math

nsize = [9000, 18000, 27000, 36000, 45000, 54000, 63000, 72000, 81000, 90000, 99000, 108000, 117000, 126000, 135000, 144000, 153000, 162000, 171000, 180000, 189000, 198000, 207000, 216000, 225000, 234000, 243000, 252000, 261000, 270000, 279000, 288000, 297000, 306000, 315000, 324000, 333000, 342000, 351000, 360000]
narch = ['cpu', 'gpu']
ntimes = ['TCPU/TGPU', 'T_M_Atoms', 'T_M_Volum']
chtimes = ['T_Che_Pot']

data = dict()
for arch in narch:
    if arch not in data:
        data[arch] = dict()
    for times in ntimes:
        if times not in data[arch]:
            data[arch][times] = list()
        for size in nsize:
            command = "cat " + arch + "/npt/" + str(size) + "/mc_" + arch + "_npt_lat.sh.o* | grep " + times + " | cut -d':' -f2 | cut -d'*' -f1"
            result=subprocess.check_output(command, shell=True)
            data[arch][times].append(float(result.decode('utf-8')) * 60 / 10)

for arch in narch:
    for times in chtimes:
        if times not in data[arch]:
            data[arch][times] = list()
        for size in nsize:
            command = "cat " + arch + "/nvt/" + str(size) + "/mc_" + arch + "_nvt_lat.sh.o* | grep " + times + " | cut -d':' -f2 | cut -d'*' -f1"
            result=subprocess.check_output(command, shell=True)
            data[arch][times].append(float(result.decode('utf-8')) * 60 / 10)

# pp = pprint.PrettyPrinter(depth=6)
# pp.pprint(data)
# print()

for times in ntimes:
    plt.plot(nsize, data['cpu'][times], label="CPU")
    plt.plot(nsize, data['gpu'][times], label="GPU")
    plt.legend(loc=2)
    plt.title(times + ' (log - log)')
    plt.ylabel("seconds per step")
    plt.yscale('log')
    plt.xlabel("Problem size (natoms)")
    plt.xscale('log')
    plt.margins(0.1, 0.1)
    plt.show()
    plt.clf()

for times in chtimes:
    plt.plot(nsize, data['cpu'][times], label="CPU")
    plt.plot(nsize, data['gpu'][times], label="GPU")
    plt.legend(loc=2)
    plt.title(times + ' (log - log)')
    plt.ylabel("seconds per step")
    plt.yscale('log')
    plt.xlabel("Problem size (natoms & insertions)")
    plt.xscale('log')
    plt.margins(0.1, 0.1)
    plt.show()
    plt.clf()

speedup = dict()
for times in ntimes:
    if times not in speedup:
        speedup[times] = [data['cpu'][times][i] / data['gpu'][times][i] for i in range(len(data['cpu'][times]))]

for times in chtimes:
    if times not in speedup:
        speedup[times] = [data['cpu'][times][i] / data['gpu'][times][i] for i in range(len(data['cpu'][times]))]

for times in ntimes:
    plt.plot(nsize, speedup[times], label=times)
    plt.legend(loc=4)
    plt.title(times)
    plt.ylabel("Speed-up")
    plt.xlabel("Problem size (natoms)")
    plt.margins(0.1, 0.1)
    plt.show()
    plt.clf()

for times in chtimes:
    plt.plot(nsize, speedup[times], label=times)
    plt.legend(loc=4)
    plt.title(times)
    plt.ylabel("Speed-up")
    plt.xlabel("Problem size (natoms & insertions)")
    plt.margins(0.1, 0.1)
    plt.show()
    plt.clf()

# pp.pprint(speedup)
# print()

print('\nMax Speed-Up')
for times in ntimes:
    print(times)
    print(max(speedup[times]))
for times in chtimes:
    print(times)
    print(max(speedup[times]))

print('\nPower law exponent')

ylog = np.zeros(len(data['cpu']['TCPU/TGPU']), dtype='float')
xlog = np.zeros(len(nsize), dtype='float')

for i in range(len(nsize)):
    xlog[i] = math.log(nsize[i])

for times in ntimes:
    plt.title(times)
    for arch in narch:
        for i in range(len(data[arch][times])):
            ylog[i] = math.log(data[arch][times][i])

        a, b = np.polyfit(xlog, ylog, 1)
        r = np.corrcoef(xlog, ylog)
        plt.plot(xlog, ylog, 'o')
        plt.ylabel("log(seconds per step)")
        plt.xlabel("log(natoms)")
        #plt.xlim(np.min(xlog) -1, np.max(xlog) +1)
        #plt.ylim(np.min(ylog) -1, np.max(ylog) +1)
        plt.plot(xlog, a * xlog + b, label = arch + ' ' + ' -> ' + 'y = {0:2.3f} x + {1:2.3f}'.format(a, b))
        plt.legend(loc=4)
        print(arch + ' ' + times + ' -> y = ' + str(a) + ' x + ' + str(b) + ' -> gamma = {0:2.3f}'.format(a))
    plt.show()
    plt.clf()

for times in chtimes:
    plt.title(times)
    for arch in narch:
        for i in range(len(data[arch][times])):
            ylog[i] = math.log(data[arch][times][i])

        a, b = np.polyfit(xlog, ylog, 1)
        r = np.corrcoef(xlog, ylog)
        plt.plot(xlog, ylog, 'o')
        plt.ylabel("log(seconds per step)")
        plt.xlabel("log(natoms & insertions)")
        #plt.xlim(np.min(xlog) -1, np.max(xlog) +1)
        #plt.ylim(np.min(ylog) -1, np.max(ylog) +1)
        plt.plot(xlog, a * xlog + b, label = arch + ' ' + ' -> ' + 'y = {0:2.3f} x + {1:2.3f}'.format(a, b))
        plt.legend(loc=4)
        print(arch + ' ' + times + ' -> y = ' + str(a) + ' x + ' + str(b) + ' -> gamma = {0:2.3f}'.format(a))
    plt.show()
    plt.clf()

print()
