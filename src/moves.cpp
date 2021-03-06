/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic
 * simulations.
 *
 * Move atoms and move volume serial code file.
 *
 * Author: adpozuelo@gmail.com
 * Version: 1.2.
 * Date: 11/2020
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../include/energy.h"
#include "../include/io.h"
#include "../include/util.h"
#include "mkl_vsl.h"

// Move atoms Metropolis Montecarlo algorithm in serial mode.
void moveAtoms(unsigned long int *ntrial, const int natoms, VSLStreamStatePtr *stream,
               double *rdmax, double *runit, double *r, const int nsp,
               int *nspps, int **itp, double *rc2, int *keyp, double *al,
               double *bl, double *cl, double *bl2, const double kt,
               double *esr, unsigned long int *naccept, double *esr_rc) {
  int ntest, iti, itj, nit, h_size = NDIM + 1;
  double rd2, rdn2, deltae, eng0, eng1;
  double *harvest = (double *)malloc(h_size * sizeof(double));
  double *rp = (double *)malloc(NDIM * sizeof(double));
  double *rdd = (double *)malloc(NDIM * sizeof(double));
  double *rddn = (double *)malloc(NDIM * sizeof(double));
  if (harvest == NULL || rp == NULL || rdd == NULL || rddn == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }

  for (int i = 0; i < natoms; ++i) {  // for every atom/particle
    (*ntrial)++;                      // update trial counter
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, *stream, h_size, harvest, 0,
                 1);                      // generate uniform random numbers
    ntest = (int)natoms * harvest[NDIM];  // set atom/particle to move

    for (int j = 0; j < NDIM; ++j) {
      // calculate new position for the particle to test
      rp[j] = r[ntest * NDIM + j] + rdmax[j] * (2 * harvest[j] - 1) / runit[j];
      // check simulation box periodic conditions
      if (rp[j] < -0.5) {
        rp[j] += 1;
      }
      if (rp[j] > 0.5) {
        rp[j] -= 1;
      }
    }

    eng0 = eng1 = 0.0;                  // energy before and after the movement
    for (int j = 0; j < natoms; ++j) {  // for every other atom/particle
      if (j == ntest) {
        continue;
      }

      for (int k = 0; k < NDIM; ++k) {
        // calculate distante between particles (before and after the movement)
        rdd[k] = r[j * NDIM + k] - r[ntest * NDIM + k];
        rddn[k] = r[j * NDIM + k] - rp[k];
      }

      // get interaction potential ids for the particles
      iti = getIatype(nsp, nspps, j);
      itj = getIatype(nsp, nspps, ntest);
      // get interaction potential id between particles
      nit = itp[iti][itj];

      // after movement
      rd2 = dist2(rdd, runit);  // distance power 2 between particles
      if (rd2 < rc2[nit]) {  // if distance is less than interaction pontential
                             // cutoff radio
        eng0 += fpot(rd2, nit, keyp, al, bl, cl,
                     bl2) - esr_rc[nit];  // get energy between particles
      }

      // before movement
      rdn2 = dist2(rddn, runit);
      if (rdn2 < rc2[nit]) {
        eng1 += fpot(rdn2, nit, keyp, al, bl, cl, bl2) - esr_rc[nit];
      }
    }

    deltae = eng1 - eng0;  // difference of energies (after and before movement)
    if (deltae < 0.0) {    // if diference is less than zero
      for (int k = 0; k < NDIM; ++k) {
        r[ntest * NDIM + k] = rp[k];  // accept the movement
      }

      *esr += deltae;  // update current iteration/step energy
      (*naccept)++;  // update number of success movements of particles in move
                     // atoms algorithm
    } else {         // else, movement not accepted by energy's difference
      double xi[1];
      vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, *stream, 1, xi, 0,
                   1);  // generate uniform random number

      if (exp(-deltae / kt) >
          xi[0]) {  // if random number is less than Boltzmann entropy
        for (int k = 0; k < NDIM; ++k) {
          r[ntest * NDIM + k] = rp[k];  // accept the movement
        }

        *esr += deltae;  // update current iteration/step energy
        (*naccept)++;    // update number of success movements of particles in
                         // move atoms algorithm
      }
    }
  }
  // Release memory
  free(harvest);
  free(rp);
  free(rdd);
  free(rddn);
}

// Move volume algorithm in selected run mode.
void moveVolume(double *esr, double *v0, double *side, double *a, double *b,
                double *c, double *runit, VSLStreamStatePtr *stream,
                const double vdmax, const char *scaling, const double pres,
                const double kt, const int natoms, unsigned long int *nvaccept,
                int **itp, double *r, double *rc2, const int nsp, int *nspps,
                int *keyp, double *al, double *bl, double *cl, double *bl2,
                double *esr_rc) {
  double esrOld = *esr;
  double v0Old = *v0;
  double *sideOld = (double *)malloc(NDIM * sizeof(double));
  double *aOld = (double *)malloc(NDIM * sizeof(double));
  double *bOld = (double *)malloc(NDIM * sizeof(double));
  double *cOld = (double *)malloc(NDIM * sizeof(double));
  double *runitOld = (double *)malloc(NDIM * sizeof(double));
  if (sideOld == NULL || aOld == NULL || bOld == NULL || cOld == NULL ||
      runitOld == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }

  // save variables before the volume movement
  for (int i = 0; i < NDIM; ++i) {
    sideOld[i] = side[i];
    aOld[i] = a[i];
    bOld[i] = b[i];
    cOld[i] = c[i];
    runitOld[i] = runit[i];
  }

  double xi[1];
  vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, *stream, 1, xi, 0,
               1);                     // generate uniform random number
  side[0] += (2 * xi[0] - 1) * vdmax;  // move X length side

  if (strcmp(scaling, "ortho") == 0) {  // is scaling is ortho
    double yzi[2];
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, *stream, 2, yzi, 0,
                 1);  // generate uniform random number
    side[1] +=
        (2 * yzi[0] - 1) * vdmax;  // move Y length side using random number
    side[2] +=
        (2 * yzi[1] - 1) * vdmax;  // move Z length side using random number
  } else {                         // is scaling is not ortho
    double factor = side[0] / sideOld[0];  // calculate factor side
    side[1] *= factor;                     // move Y length side using factor
    side[2] *= factor;                     // move Z length side using factor
  }

  *v0 = side[0] * side[1] * side[2];  // calculate new simulation box volume
  // XYZ side length of the simulation box
  a[0] = side[0];
  b[1] = side[1];
  c[2] = side[2];

  for (int i = 0; i < NDIM; ++i) {
    runit[i] = side[i];  // set new normalization units of the simulation box's
                         // length sides
  }

  // get energy after volume's movement
  *esr = energycpu(natoms, itp, r, runit, rc2, nsp, nspps, keyp, al, bl, cl,
                   bl2, esr_rc);

  double deltaEsr = *esr - esrOld;  // get difference of energies (after and
                                    // before volume's movement)
  double cond =
      exp(-(deltaEsr + pres * (*v0 - v0Old)) / kt +
          (double)natoms *
              log(*v0 / v0Old));  // acceptance criteria for volume changes

  vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, *stream, 1, xi, 0,
               1);  // generate uniform random number

  if (cond > xi[0]) {  // if random number is less than acceptance criteria for
                       // volume changes
    (*nvaccept)++;     // accept volume movement
  } else {             // if not accept volume movement
    // restore variables to original values (before volume movement)
    *esr = esrOld;
    *v0 = v0Old;
    for (int i = 0; i < NDIM; ++i) {
      side[i] = sideOld[i];
      a[i] = aOld[i];
      b[i] = bOld[i];
      c[i] = cOld[i];
      runit[i] = runitOld[i];
    }
  }
  // Release memory
  free(sideOld);
  free(aOld);
  free(bOld);
  free(cOld);
  free(runitOld);
}
