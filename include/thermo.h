/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic
 * simulations.
 *
 * Statistics header file.
 *
 * Author: adpozuelo@uoc.edu
 * Version: 1.0.
 * Date: 2018
 */

#ifndef THERMO_H_
#define THERMO_H_

#include <stdbool.h>

/**
   Generate histograms.
   @arguments:
   ensemble: simulation ensemble (nvt or npt)
   natoms: total number of atoms/particles
   esr: current iteration/step energy
   v0: simulation box volume
   eref: reference energy
   deltaeng: nvt delta grid for energy histogram
   deltar: npt delta grid for density histogram
   @return:
   etotal: total energy
   ehisto: energy histogram
   rhisto: density histogram
*/
void histograms(const char *ensemble, double *etotal,
                unsigned long long int *ehisto, const int natoms,
                const double esr, const double v0, const double eref,
                const double deltaeng, const double deltar,
                unsigned long long int *rhisto);

/**
   Write statistics to output files
   @arguments:
   ensemble: simulation ensemble (nvt or npt)
   ehisto: energy histogram
   deltaeng: nvt delta grid for energy histogram
   v0: simulation box volume
   esr: current iteration/step energy
   sideav: accumulated side average
   side: XYZ side length of the simulation box
   rhisto: density histogram
   deltar: npt delta grid for density histogram
   natoms: total number of atoms/particles
   ntrial: number of trials executed
   naccept: number of success movements of particles in move atoms algorithm
   nvaccept: number of success movements of particles in move volume algorithm
   sigma_o: potential sigma normalization value
   eps_o: potental epsilon normalization value
   @return:
   first: run boolean to control if equilibrium phase is over
   eref: reference energy
   volav: accumulated total volume
   naver: number of averages executed
   etotal: total energy
   etav: accumulated total energy average
   esav: accumulated iteration energy average
*/
void averages(bool *first, const char *ensemble, unsigned long long int *ehisto,
              const double deltaeng, double *eref, double *volav,
              const double v0, const double esr, double *sideav, double *side,
              unsigned long long int *rhisto, const double deltar, int *naver,
              double *etotal, double *etav, double *esav, const int natoms,
              const int ntrial, const int naccept, const int nvaccept,
              const double sigma_o, const double eps_o,
              const double final_sm_rate, double *vdmax, double *rdmax);

#endif
