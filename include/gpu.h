/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic
 * simulations.
 *
 * GPU functions header file.
 *
 * Author: adpozuelo@uoc.edu
 * Version: 1.0.
 * Date: 2018
 */

#ifndef GPU_H
#define GPU_H

#include "mkl_vsl.h"

/**
   Set of GPU functions:
   mode 0: Initialize GPU memory.
   mode 1: Energy of a configuration in parallel mode.
   mode 2: Move atoms Metropolis Montecarlo algorithm in parallel mode.
   mode 3: Chemical potental algorithm in parallel mode.
   mode 4: Move volume algorithm in serial mode.
   mode 5: Release GPU memory
   @arguments:
   mode: function code to execute
   natoms: total number of atoms/particles
   itp: interaction potentials between species
   r: particles positions (xyz)
   runit: normalization units of the simulation box's length sides
   rc2: cutoff radio Morse/LJ parameters power 2
   nsp: number of species
   nspps: accumulated number of species per specie
   keyp: potential's key -> 1 = Morse, 2 = Lennard Jones
   al: Morse/LJ parameters
   bl: Morse/LJ parameters
   cl: Morse/LJ parameters
   bl2: Morse/LJ parameters
   side: XYZ side length of the simulation box
   a: X side length of the simulation box
   b: Y side length of the simulation box
   c: Z side length of the simulation box
   stream: uniform random number generator stream for MC simulation (chemical
   potential excluded) vdmax: volume maximum displacement scaling: simulation
   scaling (ortho or isotr) pres: simulation box pression kt: Boltzmann constant
   rdmax: maximum trial displacements
   chpotit: chemical potential iterations (number of particles inserted for
   every specie) cudadevice: cuda device in which run the algorithm.
   @return:
   mode 1:
      the energy of a configuration
   mode 2:
      ntrial: number of trials executed
      esr: current iteration/step energy
      naccept: number of success movements of particles in move atoms algorithm
   mode 4:
      esr: current iteration/step energy
      v0: simulation box volume
      nvaccept: number of success movements of particles in move volume
   algorithm
*/

extern "C" void gpu(const int mode, const int natoms, int **itp, double *r,
                    double *runit, double *rc2, const int nsp, int *nspps,
                    int *keyp, double *al, double *bl, double *cl, double *bl2,
                    const int nitmax, const int cudadevice, int *ntrial,
                    VSLStreamStatePtr *stream, double *rdmax, const double kt,
                    double *esr, int *naccept, const int chpotit, double *v0,
                    double *side, double *a, double *b, double *c,
                    const double vdmax, const char *scaling, const double pres,
                    int *nvaccept);

#endif
