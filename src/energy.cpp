/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic
 * simulations.
 *
 * Potential/Energy serial code file.
 *
 * Author: adpozuelo@gmail.com
 * Version: 1.1.
 * Date: 11/2020
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../include/io.h"
#include "../include/util.h"
#include "mkl_vsl.h"

#define KBEV 8.6173324e-5
#define BAR2EV 0.624150932e-6
#define BAR2K 0.72429715652e-2
#define UPMAX 80.0
#define DR 0.001

// Initialize simulation box potential environment.
void initPotential(double *kt, double *pres, const double temp,
                   const char *units, double *rcut, double **rc, const int nsp,
                   double *rcut2, const double eps_o) {
  if (strcmp(units, "eV") == 0) {  // if unit is eV
    *kt = KBEV * temp;             // initialize Boltzmann constant
    *pres *= BAR2EV;               // initialize simulation box pression
  }

  if (strcmp(units, "K") == 0) {  // the same for K
    *kt = temp;
    *pres *= BAR2K;
  }

  *rcut = rc[0][0];
  for (int i = 0; i < nsp; ++i) {
    for (int j = 0; j < nsp; ++j) {
      if (rc[i][j] > *rcut)
        *rcut = rc[i][j];  // use the wider interaction cutoff radio
    }
  }
  *kt /= eps_o;            // normalize Boltzmann constant
  *rcut2 = *rcut * *rcut;  // interaction cutoff radio power 2
}

// Energy between two particles.
double fpot(const double r2, const int nit, int *keyp, double *al, double *bl,
            double *cl, double *bl2) {
  double rr, r6;

  // Morse potential (https://en.wikipedia.org/wiki/Morse_potential)
  if (keyp[nit] == 1) {
    rr = sqrt(r2);
    double expp = exp(-bl[nit] * (rr - cl[nit]));
    return al[nit] * ((1 - expp) * (1 - expp) - 1);

    // LJ potential (https://en.wikipedia.org/wiki/Lennard-Jones_potential)
  } else if (keyp[nit] == 2) {
    r6 = (bl2[nit] / r2) * (bl2[nit] / r2) * (bl2[nit] / r2);
    return 4 * al[nit] * r6 * (r6 - 1.0);

  } else {
    fputs("ERROR: interaction not implemented!\n", stderr);
    exit(1);
  }
}

// Energy of a configuration in serial mode.
double energycpu(const int natoms, int **itp, double *r, double *runit,
                 double *rc2, const int nsp, int *nspps, int *keyp, double *al,
                 double *bl, double *cl, double *bl2, double *esr_rc) {
  double rd2, eng = 0.0;
  int iti, itj, nit;
  double *rdd = (double *)malloc(NDIM * sizeof(double));
  if (rdd == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }

  for (int i = 0; i < natoms; ++i) {        // for every particle
    for (int j = i + 1; j < natoms; ++j) {  // travel the other particles

      // get interaction potential ids for the particles
      iti = getIatype(nsp, nspps, i);
      itj = getIatype(nsp, nspps, j);
      // get interaction potential id between particles
      nit = itp[iti][itj];

      for (int k = 0; k < NDIM; ++k) {
        rdd[k] = r[i * NDIM + k] -
                 r[j * NDIM + k];  // calculate distante between particles
      }

      rd2 = dist2(rdd, runit);  // distance power 2 between particles
      if (rd2 < rc2[nit]) {  // if distance is less than interaction pontetial
                             // cutoff radio
        eng += fpot(rd2, nit, keyp, al, bl, cl,
                    bl2) - esr_rc[nit];  // accumulate energy between particles
      }
    }
  }

  free(rdd);
  return eng;
}

// Chemical potental algorithm in serial mode.
void chpotentialcpu(const int chpotit, const int natoms, int **itp, double *r,
                    double *runit, double *rc2, const int nsp, int *nspps,
                    int *keyp, double *al, double *bl, double *cl, double *bl2,
                    const double kt, double *esr_rc) {
  VSLStreamStatePtr stream;  // uniform random number generator stream
  // vslNewStream(&stream, VSL_BRNG_MT19937, time(NULL)); // random mode
  vslNewStream(&stream, VSL_BRNG_MT19937, 1);  // deterministic mode

  const int h_size = chpotit * NDIM;
  double *harvest = (double *)malloc(h_size * sizeof(double));
  double *deltae = (double *)malloc(chpotit * sizeof(double));
  if (harvest == NULL || deltae == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }

  double xyz[NDIM], rdd[NDIM];
  double rd2, eng, chpot;
  int iti, itj, nit;

  const char *filename = "../results/chpotential.dat";  // output filename
  FILE *fp;
  fp = fopen(filename, "a");  // open output file in append mode

  for (int i = 0; i < nsp; ++i) {  // for every specie

    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, h_size, harvest, 0,
                 1);  // generate uniform random numbers
    iti = i;          // the id for the inserted particle

    for (int j = 0; j < chpotit; ++j) {  // chemical potential iterations

      for (int k = 0; k < NDIM; ++k) {
        xyz[k] =
            harvest[j * NDIM + k];  // get random inserted particle's position
      }

      eng = 0.0;  // energy accumulator set to zero
      for (int n = 0; n < natoms;
           ++n) {  // for every particle in the configuration

        itj =
            getIatype(nsp, nspps, n);  // the id for the configuration particle
        nit = itp[iti][itj];  // get interaction potential id between particles

        for (int k = 0; k < NDIM; ++k) {
          rdd[k] =
              xyz[k] - r[n * NDIM + k];  // calculate distante between particles
                                         // (inserted and existing)
        }

        rd2 = dist2(rdd, runit);  // distance power 2 between particles
                                  // (inserted and existing)
        if (rd2 < rc2[nit]) {  // if distance is less than interaction pontetial
                               // cutoff radio
          eng += fpot(rd2, nit, keyp, al, bl, cl,
                      bl2) - esr_rc[nit];  // accumulate energy between particles
        }
      }
      deltae[j] =
          exp(-eng /
              kt);  // energy changes related to the excess chemical potential
    }

    chpot = 0.0;
    // calculate de average value for the excess chemical potential
    for (int j = 0; j < chpotit; ++j) {
      chpot += deltae[j];
    }
    chpot /= chpotit;

    // write excess chemical potential to output file
    if (fprintf(fp, "%f\t", chpot) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filename);
      exit(1);
    }
  }

  if (fputc('\n', fp) == EOF) {
    printf("ERROR: cannot write to '%s' file!\n", filename);
    exit(1);
  }

  fclose(fp);  // close output file
  // release memory
  vslDeleteStream(&stream);
  free(harvest);
  free(deltae);
}
