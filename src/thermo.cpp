/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic
 * simulations.
 *
 * Statistics code file.
 *
 * Author: adpozuelo@gmail.com
 * Version: 2.0
 * Date: 2020
 */

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "../include/io.h"
#include "../include/util.h"

#define M_PI 3.14159265358979323846

// Generate histograms.
void histograms(const char *ensemble, double *etotal,
                unsigned long long int *ehisto, const int natoms,
                const double esr, const double v0, const double eref,
                const double deltaeng, const double deltar,
                unsigned long long int *rhisto) {
  if (strcmp(ensemble, "nvt") == 0) {  // if ensemble is nvt
    *etotal = esr;
    // get histogram's bind for the energy
    int nei = rint((*etotal - eref) / deltaeng) + NDE;
    // check integrity bounds for the energy's bind
    if (nei < 0 || nei > 2 * NDE) {
      printf("Warning: E histogram out of bounds, increase NDE up to %d!\n",
             abs(nei - NDE));
      exit(1);
    } else {
      ehisto[nei]++;  // add one to energy's bind
    }
  } else if (strcmp(ensemble, "npt") == 0) {  // if ensemble is npt
    double rhoins = natoms / v0;
    int nrho =
        abs(rint(rhoins / deltar));  // get histomgram's bind for the density

    if (nrho > NRMAX) {  // check integrity bound for the density's bind
      printf("Warning: RHO histogram out of bounds, increase NRMAX up to %d!\n",
             nrho);
      exit(1);
    } else {
      rhisto[nrho]++;  // add one to density's bind
    }
  }
}

// Write statistics to output files
void averages(bool *first, const char *ensemble, unsigned long long int *ehisto,
              const double deltaeng, double *eref, double *volav,
              const double v0, const double esr, double *sideav, double *side,
              unsigned long long int *rhisto, const double deltar, int *naver,
              double *etotal, double *etav, double *esav, const int natoms,
              const int ntrial, const unsigned long int naccept,
              const unsigned long int nvaccept, const double sigma_o,
              const double eps_o, const double final_sm_rate, double *vdmax,
              double *rdmax) {
  if (*first) {  // if equilibrium phase is over
    // print end of equilibrium phase information
    puts("*** End of equilibration ***\n");
    if (strcmp(ensemble, "nvt") == 0) {
      puts(
          "No. moves  % accept.        E_tot           "
          "E_sr\n--------------------------------------------------------------"
          "------------------\n");
    } else if (strcmp(ensemble, "npt") == 0) {
      puts(
          " No. moves  % accept.    % acc. vol.    E_tot           E_sr      "
          "Vol\n---------------------------------------------------------------"
          "-----------------\n");
    }
    *first = false;  // set equilibrium phase boolean control to false
  }
  if (strcmp(ensemble, "nvt") == 0) {  // if ensemble is nvt
    unsigned long long int esum = 0;
    for (int i = 0; i < 2 * NDE + 1; ++i) {  // total sum of energy histogram
      esum += ehisto[i];
    }
    // energy's output filename
    const char *efilename = "../results/ehisto.dat";
    FILE *ehistoFile = fopen(efilename, "w");  // open output file in write mode
    if (ehistoFile == NULL) {
      printf("ERROR: cannot access '%s' file!\n", efilename);
      exit(1);
    }
    // write energy's statistics to output file
    for (int i = 0; i < 2 * NDE + 1; ++i) {
      if (fprintf(ehistoFile, "%.7lf %.7lf\n",
                  (i - NDE + 0.5) * deltaeng * eps_o + *eref * eps_o,
                  ehisto[i] / (double)esum) == EOF) {
        printf("ERROR: cannot write to '%s' file!\n", efilename);
        exit(1);
      }
    }
    if (fputc('\n', ehistoFile) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", efilename);
      exit(1);
    }
    fclose(ehistoFile);  // close output file
  } else if (strcmp(ensemble, "npt") == 0) {  // if ensemble is npt
    *volav += v0;  // update average volume accumulator
    for (int i = 0; i < NDIM; ++i) {
      sideav[i] += side[i];  // update average side accumulator
    }
    unsigned long long int rsum = 0;
    for (int i = 0; i <= NRMAX; ++i) {
      rsum += rhisto[i];  // total sum of density histogram
    }
    // density's output filename
    const char *rfilename = "../results/rho_histo.dat";
    FILE *rhistoFile = fopen(rfilename, "w");  // open output file in write mode
    if (rhistoFile == NULL) {
      printf("ERROR: cannot access '%s' file!\n", rfilename);
      exit(1);
    }
    // write density's statistics to output file
    for (int i = 0; i <= NRMAX; ++i) {
      if (fprintf(rhistoFile, "%.7lf %.7lf\n", (i + 0.5) * deltar * sigma_o,
                  rhisto[i] / (double)rsum) == EOF) {
        printf("ERROR: cannot write to '%s' file!\n", rfilename);
        exit(1);
      }
    }
    if (fputc('\n', rhistoFile) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", rfilename);
      exit(1);
    }
    fclose(rhistoFile);  // close output file
  }
  (*naver)++;  // update number of averages executed
  *etotal = esr;
  *etav += *etotal;  // update accumulated total energy average
  *esav += esr;      // update accumulated iteration energy average
  // print statistics
  printout(true, etotal, eref, esr, ensemble, sideav, *etav, *naver, v0, *esav,
           *volav, side, natoms, ntrial, naccept, nvaccept, sigma_o, eps_o,
           final_sm_rate, vdmax, rdmax);
}
