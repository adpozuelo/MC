/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic simulations.
 *
 * Move atoms and move volume header file.
 *
 * Author: adpozuelo@uoc.edu
 * Version: 1.0.
 * Date: 2018
 */

#ifndef MOVES_H_
#define MOVES_H_

#include "mkl_vsl.h"

/**
   Move atoms Metropolis Montecarlo algorithm in serial mode.
   @arguments:
   natoms: total number of atoms/particles
   stream: uniform random number generator stream for MC simulation (chemical potential excluded)
   rdmax: maximum trial displacements
   runit: normalization units of the simulation box's length sides
   r: particles positions (xyz)
   nsp: number of species
   nspps: accumulated number of species per specie
   itp: interaction potentials between species
   rc2: cutoff radio Morse/LJ parameters power 2
   keyp: potential's key -> 1 = Morse, 2 = Lennard Jones
   al: Morse/LJ parameters
   bl: Morse/LJ parameters
   cl: Morse/LJ parameters
   bl2: Morse/LJ parameters
   kt: Boltzmann constant
   @return:
   ntrial: number of trials executed
   esr: current iteration/step energy
   naccept: number of success movements of particles in move atoms algorithm
*/
void moveAtoms(int *ntrial, const int natoms, VSLStreamStatePtr *stream, double *rdmax, double *runit, double *r, const int nsp, int *nspps, int **itp, double *rc2, int *keyp, double *al, double *bl, double *cl, double *bl2, const double kt, double *esr, int *naccept);

/**
   Move volume algorithm in serial mode.
   @arguments:
   side: XYZ side length of the simulation box
   a: X side length of the simulation box
   b: Y side length of the simulation box
   c: Z side length of the simulation box
   runit: normalization units of the simulation box's length sides
   stream: uniform random number generator stream for MC simulation (chemical potential excluded)
   vdmax: volume maximum displacement
   scaling: simulation scaling (ortho or isotr)
   pres: simulation box pression
   kt: Boltzmann constant
   natoms: total number of atoms/particles
   itp: interaction potentials between species
   r: particles positions (xyz)
   rc2: cutoff radio Morse/LJ parameters power 2
   nsp: number of species
   nspps: accumulated number of species per specie
   keyp: potential's key -> 1 = Morse, 2 = Lennard Jones
   al: Morse/LJ parameters
   bl: Morse/LJ parameters
   cl: Morse/LJ parameters
   bl2: Morse/LJ parameters
   @return:
   esr: current iteration/step energy
   v0: simulation box volume
   nvaccept: number of success movements of particles in move volume algorithm
*/
void moveVolume(double *esr, double *v0, double *side, double *a, double *b, double *c, double *runit, VSLStreamStatePtr *stream, const double vdmax, const char *scaling, const double pres, const double kt, const int natoms, int *nvaccept, int **itp, double *r, double *rc2, const int nsp, int *nspps, int *keyp, double *al, double *bl, double *cl, double *bl2);

#endif
