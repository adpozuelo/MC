/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic
 * simulations.
 *
 * GPU functions code file
 *
 * Author: adpozuelo@gmail.com
 * Version: 1.1.
 * Date: 11/2020
 */

#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" {
#include "../include/io.h"
#include "mkl_vsl.h"
}

// GPU (device code) interaction type (id) of a particle (see util.cpp for CPU
// code).
__device__ int __getIatype__(int nsp, int *nspps, int sp) {
  int i = 0;
  for (; i < nsp; ++i) {
    if (sp < nspps[i]) {
      break;
    }
  }
  return i;
}

// GPU (device code) dot/scalar product => distance (power 2) (see util.cpp for
// CPU code).
__device__ float __dist2__(float *r, float *runit) {
  for (int i = 0; i < NDIM; ++i) {
    if (r[i] > 0.5) {
      r[i] -= 1;
    }
    if (r[i] < -0.5) {
      r[i] += 1;
    }
  }
  float rd2 = 0.0;
  for (int i = 0; i < NDIM; ++i) {
    r[i] *= runit[i];
    r[i] *= r[i];
    rd2 += r[i];
  }
  return rd2;
}

// GPU (device code) energy between two particles (see potEnergy.cpp for CPU
// code).
__device__ float __fpot__(float r2, int nit, int *keyp, float *al, float *bl,
                          float *cl, float *bl2) {
  float r, r6;
  if (keyp[nit] == 1) {
    float rr = __fsqrt_rn(r2);
    float expp = __expf(-bl[nit] * (rr - cl[nit]));
    r = al[nit] * ((1 - expp) * (1 - expp) - 1.0);
  } else if (keyp[nit] == 2) {
    r6 = (bl2[nit] / r2) * (bl2[nit] / r2) * (bl2[nit] / r2);
    r = 4 * al[nit] * r6 * (r6 - 1.0);
  } else {
    asm("trap;");
  }
  return r;
}

// GPU chemical potential CUDA kernel.
__global__ void chpotKernel(float *r, float *chpot, int natoms, int nsp,
                            int *itp, float *runit, float *rc2, float *al,
                            float *bl, float *bl2, float *cl, int *nspps,
                            int *keyp, int sp, unsigned int seed, int chpotit,
                            float kt, float *esr_rc) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
  int tx = threadIdx.x;                             // block thread index

  curandState_t state;                // CuRand state
  curand_init(seed, idx, 0, &state);  // CuRand stream initialization

  __shared__ float deltae[NTHREAD];  // energy changes related to the excess
                                     // chemical potential (block vector)
  float xyz[NDIM], rdd[NDIM];
  float rd2, eng;
  int iti, itj, nit;

  if (idx < chpotit) {  // if global thread index is a valid chemical potential
                        // iteration; thus, every thread inserts one particle

    for (int i = 0; i < NDIM; ++i) {
      xyz[i] = curand_uniform(
          &state);  // each global thread index generate its own particle's
                    // position from CuRand uniform random number
    }

    iti = sp;   // the id for the inserted particle
    eng = 0.0;  // energy accumulator set to zero

    for (int n = 0; n < natoms;
         ++n) {  // for every particle in the configuration

      itj = __getIatype__(nsp, nspps,
                          n);  // the id for the configuration particle
      nit = itp[iti * nsp +
                itj];  // get interaction potential id between particles

      for (int k = 0; k < NDIM; ++k) {
        rdd[k] = xyz[k] - r[n * NDIM + k];  // calculate distante between
                                            // particles (inserted and existing)
      }

      rd2 = __dist2__(
          rdd,
          runit);  // distance power 2 between particles (inserted and existing)
      if (rd2 < rc2[nit]) {  // if distance is less than interaction pontetial
                             // cutoff radio
        eng += __fpot__(rd2, nit, keyp, al, bl, cl,
                        bl2) - esr_rc[nit];  // accumulate energy between particles
      }
    }
    deltae[tx] = __expf(
        -eng / kt);  // energy changes related to the excess chemical potential

  } else {  // if global thread index is not a valid chemical potential
            // iteration
    deltae[tx] = 0.0;  // energy changes related to the excess chemical
                       // potential set to zero
  }
  __syncthreads();  // sync block threads

  // binary reduction energy changes block vectors
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tx < s) {  // for every thread in the lower middle half of the block
      deltae[tx] +=
          deltae[tx + s];  // sum energy changes of its block index and its
                           // block index plus half the size of the block
    }
    __syncthreads();  // sync block threads
  }

  if (tx ==
      0) {  // if block thread index is zero (the first thread of the block)
    atomicAdd(chpot, deltae[0]);  // atomic (transactional) sum block energy
                                  // changes to energy accumulator
  }
}

// GPU delta energy (MoveVolume algorithm) CUDA kernel.
__global__ void deKernel(int ntest, float *r, float *eng0, float *eng1,
                         int natoms, float *rp, int nsp, int *itp, float *runit,
                         float *rc2, float *al, float *bl, float *bl2,
                         float *cl, int *nspps, int *keyp, float *esr_rc) {
  __shared__ float e0[NTHREAD],
      e1[NTHREAD];  // energy before and after particle movement (block vectors)
  int j = threadIdx.x + blockIdx.x * blockDim.x;  // global thread index
  int tx = threadIdx.x;                           // block thread index

  float rdd[NDIM], rddn[NDIM];
  float rd2;
  int iti, itj, nit;

  if (j < natoms) {    // if global thread index is a valid atom/particle
    if (j == ntest) {  // if global thread index is the particle to be moved (is
                       // the same particle)
      // set energies to zero
      e0[tx] = 0.0;
      e1[tx] = 0.0;
    }
    if (j != ntest) {  // if global thread index is not the particle to be moved
                       // (is not the same particle)
      for (int k = 0; k < NDIM; ++k) {
        // calculate distante between particles (before and after the movement)
        rdd[k] = r[j * NDIM + k] - r[ntest * NDIM + k];
        rddn[k] = r[j * NDIM + k] - rp[k];
      }
      // get interaction potential ids for the particles
      iti = __getIatype__(nsp, nspps, ntest);
      itj = __getIatype__(nsp, nspps, j);
      // get interaction potential id between particles
      nit = itp[iti * nsp + itj];

      // after movement
      rd2 = __dist2__(rdd, runit);  // distance power 2 between particles
      if (rd2 < rc2[nit]) {  // if distance is less than interaction pontetial
                             // cutoff radio
        e0[tx] = __fpot__(rd2, nit, keyp, al, bl, cl,
                          bl2) - esr_rc[nit];  // get energy between particles
      } else {  // if distance is more than interaction potential particles
                // don't interact
        e0[tx] = 0.0;  // energy set to zero
      }

      // before movement
      rd2 = __dist2__(rddn, runit);
      if (rd2 < rc2[nit]) {
        e1[tx] = __fpot__(rd2, nit, keyp, al, bl, cl, bl2) - esr_rc[nit];
      } else {
        e1[tx] = 0.0;
      }
    }
  } else {  // if global thread index is not a valid atom/particle
    // set energies to zero
    e0[tx] = 0.0;
    e1[tx] = 0.0;
  }
  __syncthreads();  // sync block threads before reduce block energies vectors

  // binary reduction energies block vectors
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tx < s) {  // for every thread in the lower middle half of the block
      e0[tx] += e0[tx + s];  // sum energies of its block index and its block
                             // index plus half the size of the block
      e1[tx] += e1[tx + s];
    }
    __syncthreads();  // sync block threads
  }

  if (tx ==
      0) {  // if block thread index is zero (the first thread of the block)
    // atomic (transactional) sum block energy to energy accumulator
    atomicAdd(eng0, e0[0]);  // energy after the particle's movement
    atomicAdd(eng1, e1[0]);  // energy before the particle's movement
  }
}

/**
       GPU (device code) vector binary reduction.
       @arguments:
       nitems: number of vector elements
       g_idata: vector to reduce
       @return:
       g_odata: result to reduce the input vector (sum of all vector's elements)
*/
__global__ void __binaryReduction__(int nitems, float *g_idata,
                                    float *g_odata) {
  __shared__ float sdata[NTHREAD];
  unsigned int tid = threadIdx.x;  // global thread index
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;  // block thread index

  if (i < nitems) {           // if global thread index is valid
    sdata[tid] = g_idata[i];  // read data from global memory
  } else {
    sdata[tid] = 0.0;
  }
  __syncthreads();  // sync threads

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {  // for every thread in the lower middle half of the block
      sdata[tid] +=
          sdata[tid + s];  // sum elements of its block index and its block
                           // index plus half the size of the block
    }
    __syncthreads();  // sync threads
  }

  if (tid ==
      0) {  // if block thread index is zero (the first thread of the block)
    // atomic (transactional) sum block energy to energy accumulator
    atomicAdd(g_odata, sdata[0]);
  }
}

// GPU configuration energy CUDA kernel
__global__ void eKernel(float *r, float *eng, int natoms, int nsp, int *itp,
                        float *runit, float *rc2, float *al, float *bl,
                        float *bl2, float *cl, int *nspps, int *keyp, float *esr_rc) {
  __shared__ float rx[NTHREAD], ry[NTHREAD], rz[NTHREAD];
  float xi, yi, zi, rd2;
  int i = threadIdx.x + blockIdx.x * blockDim.x;  // global thread index
  int tx = threadIdx.x;                           // block thread index
  int imol = i * NDIM;          // particle global memory vector position
  int mtx = tx * NDIM;          // block memory vector adjust
  int rest = natoms % NTHREAD;  // particles out of block fit
  int nbl = natoms / NTHREAD;   // particles inside block fit
  int ndmol = natoms * NDIM;    // total size of global memory vector position
  double energ = 0.0;
  float rdd[NDIM];
  int iti, itj, nit;

  if (i < natoms) {  // if particle (thread) is valid
    xi = r[imol];    // get its position from global memory
    yi = r[imol + 1];
    zi = r[imol + 2];
  }

  for (int m = 0; m <= nbl; ++m) {  // for every block
    int ml = m / nbl;
    int lim =
        (1 - ml) * NTHREAD + ml * rest;  // calculate the limit in the block fit
                                         // (particle's rest is considered)
    int mth = m * NTHREAD;               // block fit particle displacement
    int mtt = mth * NDIM + mtx;          // block fit particle position

    if (mtt <= ndmol) {  // if particle is valid
      rx[tx] = r[mtt];   // get its position inside current block
      ry[tx] = r[mtt + 1];
      rz[tx] = r[mtt + 2];
    }
    // sync block threads (all threads/particles stores its position in current
    // block's shared memory)
    __syncthreads();

    if (i < natoms) {                  // if particle (thread) is valid
      for (int j = 0; j < lim; ++j) {  // for every other particle inside the
                                       // block fit (check limit!)
        int jmth = j + mth;            // particle global index
        if (i != jmth) {  // if particle (thread) is different of particle which
                          // calculate interaction energy

          // get interaction potential ids for the particles
          iti = __getIatype__(nsp, nspps, i);
          itj = __getIatype__(nsp, nspps, jmth);
          // get interaction potential id between particles
          nit = itp[iti * nsp + itj];

          // calculate distante between particle
          rdd[0] = xi - rx[j];
          rdd[1] = yi - ry[j];
          rdd[2] = zi - rz[j];

          rd2 = __dist2__(rdd, runit);  // distance power 2 between particles
          if (rd2 < rc2[nit]) {         // if distance is less than interaction
                                        // pontetial cutoff radio
            energ += __fpot__(rd2, nit, keyp, al, bl, cl,
                              bl2) - esr_rc[nit];  // accumulate energy between particles
          }
        }
      }
    }
    __syncthreads();  // sync block threads

    if (i < natoms) {  // if particle (thread) is valid
      eng[i] = energ;  // set total energy to output vector
    }
  }
}

// Set of GPU functions
extern "C" void gpu(const int mode, const int natoms, int **itp, double *r,
                    double *runit, double *rc2, const int nsp, int *nspps,
                    int *keyp, double *al, double *bl, double *cl, double *bl2,
                    const int nitmax, const int cudadevice, int *ntrial,
                    VSLStreamStatePtr *stream, double *rdmax, const double kt,
                    double *esr, int *naccept, const int chpotit, double *v0,
                    double *side, double *a, double *b, double *c,
                    const double vdmax, const char *scaling, const double pres,
                    int *nvaccept, double *esr_rc) {
  // data in GPU memory has to be static!
  static int *itpdev, *nsppsdev, *keypdev;
  static float *rdev, *runitdev, *rc2dev, *aldev, *bldev, *bl2dev, *cldev, *esr_rc_dev;

  if (mode == 0) {  // Initialize GPU memory.

    cudaSetDevice(cudadevice);  // set cuda device
    // double precision to single precision temporal variables
    float *rf = (float *)malloc(natoms * NDIM * sizeof(float));
    int *itp_serialized = (int *)malloc(nsp * nsp * sizeof(int));
    float *runitf = (float *)malloc(NDIM * sizeof(float));
    float *rc2f = (float *)malloc(nitmax * sizeof(float));
    float *alf = (float *)malloc(nitmax * sizeof(float));
    float *blf = (float *)malloc(nitmax * sizeof(float));
    float *bl2f = (float *)malloc(nitmax * sizeof(float));
    float *clf = (float *)malloc(nitmax * sizeof(float));
    float *esr_rcf = (float *)malloc(nitmax * sizeof(float));
    if (rf == NULL || itp_serialized == NULL || runitf == NULL ||
        rc2f == NULL || alf == NULL || blf == NULL || bl2f == NULL ||
        clf == NULL || esr_rcf == NULL) {
      fputs(errorNEM, stderr);
      exit(1);
    }
    for (int i = 0; i < natoms * NDIM; ++i) {
      rf[i] = (float)r[i];  // convert particles positions from double precision
                            // to single precision
    }
    // allocate (in GPU) memory and copy particles positions from CPU to GPU
    cudaMalloc((void **)&rdev, natoms * NDIM * sizeof(float));
    cudaMemcpy(rdev, rf, natoms * NDIM * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < nsp; ++i) {
      for (int j = 0; j < nsp; ++j) {
        itp_serialized[i * nsp + j] =
            itp[i][j];  // interaction potentials between species must be
                        // serialized to take advantage of memory coalescenc
      }
    }
    // allocate (in GPU) interaction potentials between species from CPU to GPU
    cudaMalloc((void **)&itpdev, nsp * nsp * sizeof(int));
    cudaMemcpy(itpdev, itp_serialized, nsp * nsp * sizeof(int),
               cudaMemcpyHostToDevice);
    // allocate (in GPU) accumulated number of species per specie from CPU to
    // GPU
    cudaMalloc((void **)&nsppsdev, nsp * sizeof(int));
    cudaMemcpy(nsppsdev, nspps, nsp * sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < NDIM; ++i) {
      runitf[i] = (float)
          runit[i];  // convert normalization units of the simulation box's
                     // length sides from double precision to single precision
    }
    // allocate (in GPU) normalization units of the simulation box's length
    // sides from CPU to GPU
    cudaMalloc((void **)&runitdev, NDIM * sizeof(float));
    cudaMemcpy(runitdev, runitf, NDIM * sizeof(float), cudaMemcpyHostToDevice);
    // convert Morse/LJ parameters from double precision to single precision
    for (int i = 0; i < nitmax; ++i) {
      rc2f[i] = (float)rc2[i];
      alf[i] = (float)al[i];
      blf[i] = (float)bl[i];
      bl2f[i] = (float)bl2[i];
      clf[i] = (float)cl[i];
      esr_rcf[i] = (float)esr_rc[i]; // energy in rc to shift mode
    }
    // allocate (in GPU) Morse/LJ parameters from CPU to GPU
    cudaMalloc((void **)&rc2dev, nitmax * sizeof(float));
    cudaMemcpy(rc2dev, rc2f, nitmax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&aldev, nitmax * sizeof(float));
    cudaMemcpy(aldev, alf, nitmax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&bldev, nitmax * sizeof(float));
    cudaMemcpy(bldev, blf, nitmax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&bl2dev, nitmax * sizeof(float));
    cudaMemcpy(bl2dev, bl2f, nitmax * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&cldev, nitmax * sizeof(float));
    cudaMemcpy(cldev, clf, nitmax * sizeof(float), cudaMemcpyHostToDevice);
    // allocate (in GPU) energy in rc to shift mode
    cudaMalloc((void **)&esr_rc_dev, nitmax * sizeof(float));
    cudaMemcpy(esr_rc_dev, esr_rcf, nitmax * sizeof(float), cudaMemcpyHostToDevice);
    // allocate (in GPU) potential's key from CPU to GPU
    cudaMalloc((void **)&keypdev, nitmax * sizeof(int));
    cudaMemcpy(keypdev, keyp, nitmax * sizeof(int), cudaMemcpyHostToDevice);
    // release CPU memory
    free(rf);
    free(itp_serialized);
    free(runitf);
    free(rc2f);
    free(alf);
    free(blf);
    free(bl2f);
    free(clf);
    free(esr_rcf);

  } else if (mode == 1) {  // Energy of a configuration in parallel mode.

    // final energy
    float eng;
    float *edev;
    // allocate (in GPU every thread interaction energy)
    cudaMalloc((void **)&edev, natoms * sizeof(float));
    float *engdev;
    // allocate (in GPU) the energy accumulator
    cudaMalloc((void **)&engdev, sizeof(float));

    int nblock =
        natoms / NTHREAD;  // calculate the number of blocks in GPU grid
    if (natoms % NTHREAD != 0) {
      ++nblock;
    }

    // call CUDA energy kernel
    eKernel<<<nblock, NTHREAD>>>(rdev, edev, natoms, nsp, itpdev, runitdev,
                                 rc2dev, aldev, bldev, bl2dev, cldev, nsppsdev,
                                 keypdev, esr_rc_dev);
    // cudaCheckError();
    cudaMemset(
        engdev, 0,
        sizeof(
            float));  // set the energy's accumulator to zero before reduction

    // call CUDA binary reduction kernel
    __binaryReduction__<<<nblock, NTHREAD>>>(natoms, edev, engdev);
    // cudaCheckError();
    cudaMemcpy(
        &eng, engdev, sizeof(float),
        cudaMemcpyDeviceToHost);  // copy energy accumulator from GPU to CPU

    cudaFree(engdev);  // release GPU memory
    cudaFree(edev);

    *esr = eng / 2;  // return energy

  } else if (mode == 2) {  // Move atoms Metropolis Montecarlo algorithm in
                           // parallel mode.

    int ntest, h_size = NDIM + 1;
    float deltae, eng0, eng1;
    double *harvest = (double *)malloc(h_size * sizeof(double));
    double *rp = (double *)malloc(NDIM * sizeof(double));
    float *rpf = (float *)malloc(NDIM * sizeof(float));
    if (harvest == NULL || rpf == NULL || rp == NULL) {
      fputs(errorNEM, stderr);
      exit(1);
    }

    float *rpdev;
    // allocate (in GPU) memory moved particle position
    cudaMalloc((void **)&rpdev, NDIM * sizeof(float));
    float *e0dev, *e1dev;
    // allocate (in GPU) energy accumulators (after and before the particle's
    // movement)
    cudaMalloc((void **)&e0dev, sizeof(float));
    cudaMalloc((void **)&e1dev, sizeof(float));

    const int nblock = (natoms + (NTHREAD - 1)) /
                       NTHREAD;  // calculate the number of blocks in GPU grid

    for (int i = 0; i < natoms; ++i) {  // for every atom/particle

      (*ntrial)++;  // update trial counter
      vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, *stream, h_size, harvest, 0,
                   1);                      // generate uniform random numbers
      ntest = (int)natoms * harvest[NDIM];  // set atom/particle to move

      for (int j = 0; j < NDIM; ++j) {
        // calculate new position for the particle to test
        rp[j] =
            r[ntest * NDIM + j] + rdmax[j] * (2 * harvest[j] - 1) / runit[j];
        // check simulation box periodic conditions for particle to test
        if (rp[j] < -0.5) {
          rp[j] += 1;
        }
        if (rp[j] > 0.5) {
          rp[j] -= 1;
        }
        rpf[j] =
            (float)rp[j];  // convert particle's positions (after the movement)
                           // from double precision to single precision
      }

      // copy particles positions (after the movement) from CPU to GPU
      cudaMemcpy(rpdev, rpf, NDIM * sizeof(float), cudaMemcpyHostToDevice);
      // set energies accumulators to zero
      cudaMemset(e0dev, 0, sizeof(float));
      cudaMemset(e1dev, 0, sizeof(float));

      // call CUDA difference of energies kernel
      deKernel<<<nblock, NTHREAD>>>(ntest, rdev, e0dev, e1dev, natoms, rpdev,
                                    nsp, itpdev, runitdev, rc2dev, aldev, bldev,
                                    bl2dev, cldev, nsppsdev, keypdev, esr_rc_dev);
      // cudaCheckError();
      // copy energy accumulators (energy after and before the movement) from
      // GPU to CPU
      cudaMemcpy(&eng0, e0dev, sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(&eng1, e1dev, sizeof(float), cudaMemcpyDeviceToHost);

      deltae =
          eng1 - eng0;  // difference of energies (after and before movement)

      if (deltae < 0.0) {  // if diference is less than zero
        for (int k = 0; k < NDIM; ++k) {
          r[ntest * NDIM + k] = rp[k];  // accept the movement
        }
        // accept the movement inside GPU (avoid recurrent and duplicated
        // particle positions copy between CPU and GPU)
        cudaMemcpy(rdev + ntest * NDIM, rpf, NDIM * sizeof(float),
                   cudaMemcpyHostToDevice);  // only copy the position of the
                                             // moved particle
        *esr += deltae;  // update current iteration/step energy
        (*naccept)++;    // update number of success movements of particles in
                         // move atoms algorithm

      } else {  // else, movement not accepted by energy's difference
        double xi[1];
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, *stream, 1, xi, 0,
                     1);  // generate uniform random number
        if (exp(-deltae / kt) >
            xi[0]) {  // if random number is less than Boltzmann entropy
          for (int k = 0; k < NDIM; ++k) {
            r[ntest * NDIM + k] = rp[k];  // accept the movement
          }
          // accept the movement inside GPU (avoid recurrent and duplicated
          // particle positions copy between CPU and GPU)
          cudaMemcpy(rdev + ntest * NDIM, rpf, NDIM * sizeof(float),
                     cudaMemcpyHostToDevice);  // only copy the position of the
                                               // moved particle
          *esr += deltae;  // update current iteration/step energy
          (*naccept)++;    // update number of success movements of particles in
                           // move atoms algorithm
        }
      }
    }

    // release CPU and GPU memory
    cudaFree(e0dev);
    cudaFree(e1dev);
    cudaFree(rpdev);
    free(harvest);
    free(rp);
    free(rpf);

  } else if (mode == 3) {  // Chemical potental algorithm in parallel mode.

    float chpot;  // final chemical potential
    float *edev;
    cudaMalloc((void **)&edev, sizeof(float));  // GPU accumulator

    const int nblock = (chpotit + (NTHREAD - 1)) /
                       NTHREAD;  // calculate the number of blocks in GPU grid

    const char *filename = "../results/chpotential.dat";  // output filename
    FILE *fp;
    fp = fopen(filename, "a");  // open output file in append mode

    static int counter = 1;  // random number generator deterministic mode

    for (int i = 0; i < nsp; ++i) {  // for every specie

      cudaMemset(edev, 0,
                 sizeof(float));  // set excess chemical potential to zero

      // chpotKernel<<<nblock, NTHREAD>>>(rdev, edev, natoms, nsp, itpdev,
      // runitdev, rc2dev, aldev, bldev, bl2dev, cldev, nsppsdev, keypdev, i,
      // time(NULL), chpotit, (float)kt, esr_rc_dev); // random mode
      chpotKernel<<<nblock, NTHREAD>>>(
          rdev, edev, natoms, nsp, itpdev, runitdev, rc2dev, aldev, bldev,
          bl2dev, cldev, nsppsdev, keypdev, i, 1 + counter, chpotit,
          (float)kt, esr_rc_dev);  // deterministic mode
      // cudaCheckError();

      counter++;  // random number generator deterministic mode

      // copy excess chemical potential accumulator from GPU to CPU
      cudaMemcpy(&chpot, edev, sizeof(float), cudaMemcpyDeviceToHost);
      chpot /= chpotit;  // get average of excess chemical potential

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

    // release GPU memory and close output file
    cudaFree(edev);
    fclose(fp);

  } else if (mode == 4) {  // Move volume algorithm in serial mode.

    double esrOld = *esr;
    double v0Old = *v0;
    double *sideOld = (double *)malloc(NDIM * sizeof(double));
    double *aOld = (double *)malloc(NDIM * sizeof(double));
    double *bOld = (double *)malloc(NDIM * sizeof(double));
    double *cOld = (double *)malloc(NDIM * sizeof(double));
    double *runitOld = (double *)malloc(NDIM * sizeof(double));
    float *runitf = (float *)malloc(NDIM * sizeof(float));
    if (sideOld == NULL || aOld == NULL || bOld == NULL || cOld == NULL ||
        runitOld == NULL || runitf == NULL) {
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
      runit[i] = side[i];  // set new normalization units of the simulation
                           // box's length sides
    }

    for (int i = 0; i < NDIM; ++i) {
      runitf[i] = (float)
          runit[i];  // convert normalization units of the simulation box's
                     // length sides from double precision to single precision
    }
    // allocate (in GPU) normalization units of the simulation box's length
    // sides from CPU to GPU
    cudaMemcpy(runitdev, runitf, NDIM * sizeof(float), cudaMemcpyHostToDevice);

    float eng;
    float *edev;
    cudaMalloc((void **)&edev,
               natoms * sizeof(float));  // allocate (in GPU) the particle's
                                         // interaction energy vector
    float *engdev;
    cudaMalloc((void **)&engdev,
               sizeof(float));  // allocate (in GPU) the energy accumulator

    int nblock =
        natoms / NTHREAD;  // calculate the number of blocks in GPU grid
    if (natoms % NTHREAD != 0) {
      ++nblock;
    }

    // call CUDA energy kernel
    eKernel<<<nblock, NTHREAD>>>(rdev, edev, natoms, nsp, itpdev, runitdev,
                                 rc2dev, aldev, bldev, bl2dev, cldev, nsppsdev,
                                 keypdev, esr_rc_dev);
    // cudaCheckError();
    cudaMemset(engdev, 0,
               sizeof(float));  // set the energy's accumulator to zero

    // call CUDA binary reduction kernel
    __binaryReduction__<<<nblock, NTHREAD>>>(natoms, edev, engdev);
    // cudaCheckError();
    // copy energy accumulator from GPU to CPU
    cudaMemcpy(&eng, engdev, sizeof(float), cudaMemcpyDeviceToHost);

    // rrelease GPU memory
    cudaFree(engdev);
    cudaFree(edev);

    *esr = eng / 2;  // return energy

    double deltaEsr = *esr - esrOld;  // get difference of energies (after and
                                      // before volume's movement)
    double cond =
        exp(-(deltaEsr + pres * (*v0 - v0Old)) / kt +
            (double)natoms *
                log(*v0 / v0Old));  // acceptance criteria for volume changes

    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, *stream, 1, xi, 0,
                 1);  // generate uniform random number

    if (cond > xi[0]) {  // if random number is less than acceptance criteria
                         // for volume changes
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
      for (int i = 0; i < NDIM; ++i) {
        runitf[i] = (float)runitOld[i];  // convert normalization units of the
                                         // simulation box's length sides from
                                         // double precision to single precision
      }
      // allocate (in GPU) normalization units of the simulation box's length
      // sides from CPU to GPU
      cudaMemcpy(runitdev, runitf, NDIM * sizeof(float),
                 cudaMemcpyHostToDevice);
    }

    // release CPU memory
    free(sideOld);
    free(aOld);
    free(bOld);
    free(cOld);
    free(runitOld);
    free(runitf);

  } else if (mode == 5) {  // Release GPU memory

    cudaFree(itpdev);
    cudaFree(nsppsdev);
    cudaFree(keypdev);
    cudaFree(rdev);
    cudaFree(runitdev);
    cudaFree(rc2dev);
    cudaFree(aldev);
    cudaFree(bldev);
    cudaFree(bl2dev);
    cudaFree(cldev);
    cudaFree(esr_rc_dev);

  } else {
    fputs("ERROR: Incorrect GPU code!\n", stderr);
    exit(1);
  }
}
