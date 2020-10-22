/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic
 * simulations.
 *
 * Potential/Energy header file.
 *
 * Author: adpozuelo@uoc.edu
 * Version: 1.0.
 * Date: 2018
 */

#ifndef ENERGY_H
#define ENERGY_H

/**
   Initialize simulation box potential environment.
   @arguments:
   temp: simulation box temperature
   units: potential units
   rc: cutoff radio Morse/LJ parameters
   nsp: number of species
   eps_o: potental epsilon normalization value
   @return:
   kt: Boltzmann constant
   pres: simulation box pression
   rcut: interaction cutoff radio
   rcut2: interaction cutoff radio power 2
*/
void initPotential(double *kt, double *pres, const double temp,
                   const char *units, double *rcut, double **rc, const int nsp,
                   double *rcut2, const double eps_o);

/**
   Energy between two particles.
   @arguments:
   r2: distance (power 2) between two particles
   nit: interaction potential key between two particles
   keyp: potential's key -> 1 = Morse, 2 = Lennard Jones
   al: Morse/LJ parameters
   bl: Morse/LJ parameters
   cl: Morse/LJ parameters
   bl2: Morse/LJ parameters
   @return:
   the energy between two particles
*/
double fpot(const double r2, const int nit, int *keyp, double *al, double *bl,
            double *cl, double *bl2);

/**
   Energy of a configuration in serial mode.
   @arguments:
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
   @return:
   the energy of a configuration
*/
double energycpu(const int natoms, int **itp, double *r, double *runit,
                 double *rc2, const int nsp, int *nspps, int *keyp, double *al,
                 double *bl, double *cl, double *bl2);

/**
   Chemical potental algorithm in serial mode.
   @arguments:
   chpotit: chemical potential iterations (number of particles inserted for
   every specie) natoms: total number of atoms/particles itp: interaction
   potentials between species r: particles positions (xyz) runit: normalization
   units of the simulation box's length sides rc2: cutoff radio Morse/LJ
   parameters power 2 nsp: number of species nspps: accumulated number of
   species per specie keyp: potential's key -> 1 = Morse, 2 = Lennard Jones al:
   Morse/LJ parameters bl: Morse/LJ parameters cl: Morse/LJ parameters bl2:
   Morse/LJ parameters kt: Boltzmann constant
*/
void chpotentialcpu(const int chpotit, const int natoms, int **itp, double *r,
                    double *runit, double *rc2, const int nsp, int *nspps,
                    int *keyp, double *al, double *bl, double *cl, double *bl2,
                    const double kt);

#endif
