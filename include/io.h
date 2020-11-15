/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic
 * simulations.
 *
 * Input/Output header file.
 *
 * Author: adpozuelo@gmail.com
 * Version: 1.2.
 * Date: 11/2020
 */

#ifndef IO_H
#define IO_H

#include <stdbool.h>
#include <stdio.h>

#define MAX_LINE_SIZE 256  // Maximum size (in chars) of a line
#define errorNEM "ERROR: Can't allocate memory!\n"  // Error not enough memory
#define NDE 50000                                   // Energy histogram's bound
#define NRMAX 10000                                 // Density histogram's bound
#define NDIM 3       // Number of dimensions (XYZ)
#define NTHREAD 128  // Number of CUDA threads per block

// CUDA kernel's check error macro
#define cudaCheckError()                                         \
  {                                                              \
    cudaError_t e = cudaGetLastError();                          \
    if (e != cudaSuccess) {                                      \
      printf("\nCuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                             \
      exit(0);                                                   \
    }                                                            \
  }

/**
   Read a line from a file.
   @arguments:
   file: FILE from read the line
   filename: filename of the FILE
   @return:
   line: readed line from FILE with filename
 */
void readLine(FILE *file, const char *filename, char *line);

/**
   Read system.dat input file.
   @return:
   initcf: configuration model (dlp or lattice)
   nsp: number of species
   nitmax: max number of interactions between species
   atoms: specie's chemical character representation
   nspps: accumulated number of species per specie
   natoms: total number of atoms/particles
   units: potential units
   keyp: potential's key -> 1 = Morse, 2 = Lennard Jones
   cl: Morse/LJ parameters
   bl2: Morse/LJ parameters
   al: Morse/LJ parameters
   bl: Morse/LJ parameters
   rc2: cutoff radio Morse/LJ parameters power 2
   rc: cutoff radio Morse/LJ parameters
   itp: interaction potentials between species
   rho: simulation box density
   sigma_o: potential sigma normalization value
   eps_o: potental epsilon normalization value
 */
void readSystemDatFile(char **initcf, int *nsp, int *nitmax, char ***atoms,
                       int **nspps, int *natoms, char **units, int **keyp,
                       double **cl, double **bl2, double **al, double **bl,
                       double **rc2, double ***rc, int ***itp, double *rho,
                       double *sigma_o, double *eps_o);

/**
   Read CONFIG input file.
   @arguments:
   nsp: number of species
   natoms: total number of atoms/particles
   nspps: accumulated number of species per specie
   atoms: specie's chemical character representation
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
void readConfigFile(double **a, double **b, double **c, const int nsp,
                    const int natoms, int **ntype, double **r, int *nspps,
                    char **atoms, double **side, double **runit, double *v0,
                    const double sigma_o);

/**
   Read runMC.dat input file.
   @arguments:
   eps_o: potental epsilon normalization value
   sigma_o: potential sigma normalization value
   @return:
   ensemble: simulation ensemble (nvt or npt)
   nstep: number of total simulation's steps
   nequil: number of equilibrium phase steps
   nb: every steps averages will be executed
   wc: every steps configuration file will be writed
   rdmax: maximum trial displacements
   temp: simulation box temperature
   deltaeng: nvt delta grid for energy histogram
   ehisto: energy histogram
   vdmax: volume maximum displacement
   scaling: simulation scaling (ortho or isotr)
   pres: simulation box pression
   deltar: npt energy maximum displacement
   sideav: accumulated side average
   rhisto: density histogram
   chpotnb: every steps chemical potential will be executed
   chpotit: chemical potential iterations (number of particles inserted for
   every specie)
 */
void readRunMCFile(char **ensemble, int *nstep, int *nequil, int *nb, int *wc,
                   double **rdmax, double *temp, double *deltaeng,
                   unsigned long long int **ehisto, double *vdmax,
                   char **scaling, double *pres, double *deltar,
                   double **sideav, unsigned long long int **rhisto,
                   const double eps_o, const double sigma_o, int *chpotnb,
                   int *chpotit, double *final_sm_rate, int *shift);

/**
   Initialize output files
   @arguments:
   ensemble: simulation ensemble (nvt or npt)
   nsp: number of species
   atoms: specie's chemical character representation
 */
void initOutputFiles(const char *ensemble, const int nsp, char **atoms);

/**
   Print average and/or instant results and write them to output files.
   @arguments:
   inst: write or not to output files
   esr: current iteration/step energy
   ensemble: simulation ensemble (nvt or npt)
   sideav: accumulated side average
   etav: accumulated total energy average
   naver: number of averages executed
   v0: simulation box volume
   esav: accumulated iteration energy average
   volav: accumulated total volume
   side: XYZ side length of the simulation box
   natoms: total number of atoms/particles
   ntrial: number of trials executed
   naccept: number of success movements of particles in move atoms algorithm
   nvaccept: number of success movements of particles in move volume algorithm
   sigma_o: potential sigma normalization value
   eps_o: potental epsilon normalization value
   @return:
   etotal: total energy
   eref: reference energy

 */
void printout(const bool inst, double *etotal, double *eref, const double esr,
              const char *ensemble, double *sideav, const double etav,
              const unsigned long int naver, const double v0, const double esav,
              const double volav, double *side, const int natoms,
              const unsigned long int ntrial, const unsigned long int naccept,
              const unsigned long int nvaccept, const double sigma_o,
              const double eps_o, const double final_sm_rate, double *vdmax,
              double *rdmax);

/**
   Write configuration to conf.xyz output file
   @arguments:
   natoms: total number of atoms/particles
   nspps: accumulated number of species per specie
   atoms: specie's chemical character representation
   r: particles positions (xyz)
   runit: normalization units of the simulation box's length sides
 */
void writeConf(const int natoms, int *nspps, char **atoms, double *r,
               double *runit);

#endif
