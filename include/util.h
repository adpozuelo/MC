/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic simulations.
 *
 * Utility functions header file.
 *
 * Author: adpozuelo@uoc.edu
 * Version: 1.0.
 * Date: 2018
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "mkl_vsl.h"

/**
   Dot/Scalar product => distance (power 2).
   @arguments:
   r: distance between two particles
   runit: normalization units of the simulation box's length sides
   @return:
   the distance (power 2) between two particles
 */
double dist2(double *r, double *runit);

/**
   Interaction type (id) of a particle.
   @arguments:
   nsp: number of species
   nspps: accumulated number of species per specie
   sp: particle index
   @return:
   the index of the interaction type of a particle
 */
int getIatype(const int nsp, int *nspps, const int sp);

/**
   Create a FCC (Face Center Cubic) particle lattice configuration from scratch
   @arguments:
   rho: simulation box density
   nsp: number of species
   natoms: total number of atoms/particles
   nspps: accumulated number of species per specie
   atoms: specie's chemical character representation
   stream: uniform random number generator stream for MC simulation (chemical potential excluded)
   sigma_o: potential sigma normalization value
   @return:
   a: X side length of the simulation box
   b: Y side length of the simulation box
   c: Z side length of the simulation box
   ntype: number of species per specie
   r: particles positions (xyz)
   side: XYZ side length of the simulation box
   runit: normalization units of the simulation box's length sides
   v0: simulation box volume
 */
void initLattice(const double rho, double **a, double **b, double **c, const int nsp, const int natoms, int **ntype, double **r, int *nspps, char **atoms, double **side, double **runit, double *v0, VSLStreamStatePtr *stream, const double sigma_o);

#endif /* UTIL_H_ */
