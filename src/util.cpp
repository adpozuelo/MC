/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic
 * simulations.
 *
 * Utility functions code file.
 *
 * Author: adpozuelo@gmail.com
 * Version: 2.0
 * Date: 2020
 */

#include <math.h>
#include <stdlib.h>

#include "../include/io.h"
#include "mkl_vsl.h"

// Dot/Scalar product => distance (power 2).
double dist2(double *r, double *runit) {
  // Check simulation box periodic conditions
  for (int i = 0; i < NDIM; ++i) {
    if (r[i] > 0.5) {
      r[i] -= 1;
    }
    if (r[i] < -0.5) {
      r[i] += 1;
    }
  }
  double rd2 = 0.0;
  // Dot/Scalar product
  for (int i = 0; i < NDIM; ++i) {
    r[i] *= runit[i];  // Denormalized!
    r[i] *= r[i];      // power 2!
    rd2 += r[i];
  }
  return rd2;  // return denormalized distance power 2
}

// Interaction type (id) of a particle.
int getIatype(const int nsp, int *nspps, const int sp) {
  int i = 0;
  for (; i < nsp; ++i) {
    if (sp < nspps[i]) {  // if particle belong to this specie
      break;
    }
  }
  return i;  // return the specie id of the particle
}

// Create a FCC (Face Center Cubic) particle lattice configuration from scratch
void initLattice(const double rho, double **a, double **b, double **c,
                 const int nsp, const int natoms, int **ntype, double **r,
                 int *nspps, char **atoms, double **side, double **runit,
                 double *v0, VSLStreamStatePtr *stream, const double sigma_o) {
  double ratom[4][3];
  *a = (double *)calloc(NDIM, sizeof(double));
  *b = (double *)calloc(NDIM, sizeof(double));
  *c = (double *)calloc(NDIM, sizeof(double));
  if (*a == NULL || *b == NULL || *c == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  // box length depends of the density
  double boxl = pow((double)natoms / rho, 0.33333333333);
  // normalize box length in reference of potential parameters
  (*a)[0] = (*b)[1] = (*c)[2] = boxl / sigma_o;
  // Check ortho-rombic condition
  if ((*a)[1] * (*a)[1] + (*a)[2] * (*a)[2] + (*b)[0] * (*b)[0] +
          (*b)[2] * (*b)[2] + (*c)[0] * (*c)[0] + (*c)[1] * (*c)[1] >
      1.0e-7) {
    fputs("ERROR: Non orthorombic cells not implemented!\n", stderr);
    exit(1);
  }
  *r = (double *)malloc(natoms * NDIM * sizeof(double));
  if (*r == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  // number of FCC individual cells (4 individual nodes per cell)
  int mside = (int)ceil(pow((double)natoms / 4.0, 0.333333));
  double boxfc2 = boxl / mside;  // side's length of the individual cells
  for (int i = 0; i < NDIM; ++i) {
    // initial particle position
    ratom[0][i] = 0;
    ratom[1][i] = boxfc2 / 2;
    ratom[2][i] = boxfc2 / 2;
    ratom[3][i] = boxfc2 / 2;
  }
  ratom[1][0] = 0;
  ratom[2][1] = 0;
  ratom[3][2] = 0;
  int m = 0;
  for (int i = 0; i < mside; ++i) {
    for (int j = 0; j < mside; ++j) {
      // for every side (XYZ) particle's positions displacement units
      for (int k = 0; k < mside; ++k) {
        double despx = i * boxfc2;
        double despy = j * boxfc2;
        double despz = k * boxfc2;
        for (int l = 0; l < 4; ++l) {  // for every individual node per cell
          if (m < 3 * natoms) {
            // set nodes positions for the FCC network
            (*r)[m] = ratom[l][0] + despx - boxl / 2;
            (*r)[m + 1] = ratom[l][1] + despy - boxl / 2;
            (*r)[m + 2] = ratom[l][2] + despz - boxl / 2;
            m += 3;
          }
        }
      }
    }
  }
  // shuffle iterations
  int riterations = 2;
  // tmp particle positions
  double xtmp, ytmp, ztmp;
  // random positions particle A
  int *random1 = (int *)malloc(natoms * sizeof(int));
  // random positions particle B
  int *random2 = (int *)malloc(natoms * sizeof(int));
  if (random1 == NULL || random2 == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  // shuffle particles inside the simulation box
  for (int p = 0; p < riterations; ++p) {
    // Generate random positions for particles A and B
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, *stream, natoms, random1, 0,
                 natoms);
    viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, *stream, natoms, random2, 0,
                 natoms);
    // for natoms iterations
    for (int i = 0; i < natoms; ++i) {
      // particle A position go to tmp particle position
      xtmp = (*r)[random1[i] * NDIM];
      ytmp = (*r)[random1[i] * NDIM + 1];
      ztmp = (*r)[random1[i] * NDIM + 2];
      // particle B position go to particle A position
      (*r)[random1[i] * NDIM] = (*r)[random2[i] * NDIM];
      (*r)[random1[i] * NDIM + 1] = (*r)[random2[i] * NDIM + 1];
      (*r)[random1[i] * NDIM + 2] = (*r)[random2[i] * NDIM + 2];
      // tmp particle (initial particle A) go to particle B position
      (*r)[random2[i] * NDIM] = xtmp;
      (*r)[random2[i] * NDIM + 1] = ytmp;
      (*r)[random2[i] * NDIM + 2] = ztmp;
    }
  }
  free(random1);  // release memory
  free(random2);
  *ntype = (int *)calloc(nsp, sizeof(int));
  if (*ntype == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  for (int i = 0; i < nsp; ++i) {
    (*ntype)[i] = nspps[i];  // set number of species per specie
  }
  for (int i = 1; i < nsp; ++i) {
    for (int j = 0; j < i; ++j) {
      nspps[i] +=
          nspps[j];  // create the accumulated number of species per specie
    }
  }
  *side = (double *)malloc(NDIM * sizeof(double));
  *runit = (double *)malloc(NDIM * sizeof(double));
  if (*side == NULL || *runit == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  // linearize the sides length of the simulation box
  (*side)[0] = (*a)[0];
  (*side)[1] = (*b)[1];
  (*side)[2] = (*c)[2];
  double *rd = (double *)malloc(NDIM * sizeof(double));
  if (rd == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  // dot/scalar product => distance of the box sides
  double rdd = 0.0;
  for (int i = 0; i < NDIM; ++i) {
    rd[i] = (*a)[i] * (*a)[i];
    rdd += rd[i];
  }
  // normalization units of the simulation box's length X side
  (*runit)[0] = sqrt(rdd);
  rdd = 0.0;
  for (int i = 0; i < NDIM; ++i) {
    rd[i] = (*b)[i] * (*b)[i];
    rdd += rd[i];
  }
  // normalization units of the simulation box's length Y side
  (*runit)[1] = sqrt(rdd);
  rdd = 0.0;
  for (int i = 0; i < NDIM; ++i) {
    rd[i] = (*c)[i] * (*c)[i];
    rdd += rd[i];
  }
  // normalization units of the simulation box's length Z side
  (*runit)[2] = sqrt(rdd);
  free(rd);
  // calculate the volume of the simulation box
  *v0 = (*a)[0] * (*b)[1] * (*c)[2] + (*a)[1] * (*b)[2] * (*c)[0] +
        (*a)[2] * (*b)[0] * (*c)[1] - (*a)[2] * (*b)[1] * (*c)[0] -
        (*a)[1] * (*b)[0] * (*c)[2] - (*a)[0] * (*b)[2] * (*c)[1];
  for (int i = 0; i < NDIM; ++i) {
    (*side)[i] = (*runit)[i];
  }
  // normalize particle's positions (XYZ) in reference to both potential
  // parameters and units of the simulation box's length (XYZ) side
  for (int i = 0; i < natoms; ++i) {
    for (int j = 0; j < NDIM; ++j) {
      (*r)[i * NDIM + j] /= sigma_o;
      (*r)[i * NDIM + j] /= (*runit)[j];
    }
  }
}
