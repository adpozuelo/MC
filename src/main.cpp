/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic
 * simulations.
 *
 * At present simulates bulk systems composed of soft spherical particles
 * without charges. Implemented ensembles: NVT and NPT. Chemical potential
 * implemented only for NVT ensemble.
 *
 * Input data files (see the files for parameter specifications):
 *          system.dat : contains description of the system to be simulated.
 *          runMC.dat  : contains specific parameters that control the run.
 *          CONFIG     : initial configuration in DLPOLY 2 format (to be
 * generalized).
 *
 * Output files:
 *           thermoaver.dat  : thermodynamic averages.
 *           thermoins.dat   : instantaneous thermodynamic quantities.
 *           gmix.dat        : pair correlation functions.
 *           chpotential.dat : chemical potential.
 *           conf.xyz        : initial and trajectory configuration in VMD
 * format.
 *
 * Program units:
 *        Energy: "eV" electronVolts.
 *                 "K" Lennard-Jones (Energy in Kelvin).
 *        Distance: Angstrom (internal units in box length).
 *
 * Author: adpozuelo@uoc.edu
 * Version: 1.0.
 * Date: 2018
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/energy.h"
#include "../include/gpu.h"
#include "../include/io.h"
#include "../include/moves.h"
#include "../include/thermo.h"
#include "../include/util.h"
#include "mkl_vsl.h"

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 3) {  // Input arguments control
    puts(
        "Usage: mc.exe mode [cudaDevice]\n\nmode:\nserial -> CPU\ngpu -> "
        "NVIDIA GPU\n\n[cudaDevice]:\nInteger from 0 to 7 (default = 0)\n");
    exit(1);
  }

  char *runMode;  // Get run mode (serial or parallel) from input arguments
  if (strcmp(argv[1], "serial") != 0 && strcmp(argv[1], "gpu") != 0) {
    printf("ERROR: '%s' not supported!\n", argv[1]);
    puts(
        "Usage: mc.exe mode [cudaDevice]\n\nmode:\nserial -> CPU\ngpu -> "
        "NVIDIA GPU\n\n[cudaDevice]:\nInteger from 0 to 7 (default = 0)\n");
    exit(1);
  } else {
    runMode = (char *)malloc(strlen(argv[1]) * sizeof(char));
    if (runMode == NULL) {
      fputs(errorNEM, stderr);
      exit(1);
    }
    strcpy(runMode, argv[1]);
  }

  int cudadevice = 0;                 // Default cuda device
  if (strcmp(argv[1], "gpu") == 0) {  // Get cuda device from input arguments
    if (argv[2] != NULL) {
      cudadevice = atoi(argv[2]);
      if (cudadevice < 0 || cudadevice > 7) {
        printf(
            "ERROR: '%d' is not a valid cudaDevice integer (from 0 to 7 gpus "
            "supported)\n!",
            cudadevice);
        puts(
            "Usage: mc.exe mode [cudaDevice]\n\nmode:\nserial -> CPU\ngpu -> "
            "NVIDIA GPU\n\n[cudaDevice]:\nInteger from 0 to 7 (default = 0)\n");
        exit(1);
      }
    }
  } else if (strcmp(argv[1], "serial") == 0) {
    if (argv[2] != NULL) {
      puts("ERROR: [cudaDevice] not supported in serial mode!\n");
      puts(
          "Usage: mc.exe mode [cudaDevice]\n\nmode:\nserial -> CPU\ngpu -> "
          "NVIDIA GPU\n\n[cudaDevice]:\nInteger from 0 to 7 (default = 0)\n");
      exit(1);
    }
  }

  clock_t begin = clock();  // General time control begin
  bool firstRun =
      true;  // First run boolean to control if equilibrium phase is over
  /**
     nsp: number of species
     nitmax: max number of interactions between species
     natoms: total number of atoms/particles
     naccept: number of success movements of particles in move atoms algorithm
     nvaccept: number of success movements of particles in move volume algorithm
     ntrial: number of trials executed
     naver: number of averages executed
     nstep: number of total simulation's steps
     nequil: number of equilibrium phase steps
     nb: every steps averages will be executed
     wc: every steps configuration file will be writed
     chpotnb: every steps chemical potential will be executed
     chpotit: chemical potential iterations (number of particles inserted for
     every specie)
   */
  int nsp, nitmax, natoms, naccept, nvaccept, ntrial, naver, nstep, nequil, nb,
      wc, chpotnb, chpotit;
  natoms = naccept = nvaccept = ntrial = naver =
      0;  // Accumulators initialized to zero
  /**
     nspps: accumulated number of species per specie
     keyp: potential's key -> 1 = Morse, 2 = Lennard Jones
     ntype: number of species per specie
   */
  int *nspps, *keyp, *ntype;
  /**
     itp: interaction potentials between species
   */
  int **itp;
  /**
     initcf: configuration model (dlp or lattice)
     units: potential units
     ensemble: simulation ensemble (nvt or npt)
     scaling: simulation scaling (ortho or isotr)
   */
  char *initcf, *units, *ensemble, *scaling;
  /**
     atoms: specie's chemical character representation
   */
  char **atoms;
  /**
     v0: simulation box volume
     temp: simulation box temperature
     deltaeng: nvt delta grid for energy histogram
     vdmax: volume maximum displacement
     pres: simulation box pression
     deltar: npt delta grid for density histogram
     etotal: total energy
     etav: accumulated total energy average
     esav: accumulated iteration energy average
     kt: Boltzmann constant
     rcut: interaction cutoff radio
     rcut2: interaction cutoff radio power 2
     esr: current iteration/step energy
     eref: reference energy
     volav: accumulated total volume
     rho: simulation box density
     sigma_o: potential sigma normalization value
     eps_o: potental epsilon normalization value
     final_sm_rate: final success movements rate
   */
  double v0, temp, deltaeng, vdmax, pres, deltar, etotal, etav, esav, kt, rcut,
      rcut2, esr, eref, volav, rho, sigma_o, eps_o, final_sm_rate;
  etotal = etav = esav = eref = 0.0;  // Accumulators initialized to zero
  /**
     cl: Morse/LJ parameters
     bl2: Morse/LJ parameters
     al: Morse/LJ parameters
     bl: Morse/LJ parameters
     rc2: cutoff radio Morse/LJ parameters power 2
     a: X side length of the simulation box
     b: Y side length of the simulation box
     c: Z side length of the simulation box
     side: XYZ side length of the simulation box
     runit: normalization units of the simulation box's length sides
     rdmax: maximum trial displacements
     r: particles positions (xyz)
     sideav: accumulated side average
   */
  double *cl, *bl2, *al, *bl, *rc2, *a, *b, *c, *side, *runit, *rdmax, *r,
      *sideav;
  /**
     rc: cutoff radio Morse/LJ parameters
   */
  double **rc;
  /**
     ehisto: energy histogram
     rhisto: density histogram
   */
  unsigned long long int *ehisto, *rhisto;
  /**
     stream: uniform random number generator stream for MC simulation (chemical
     potential excluded)
   */
  VSLStreamStatePtr stream;

  // vslNewStream(&stream, VSL_BRNG_MT19937, time(NULL)); // Random mode
  vslNewStream(&stream, VSL_BRNG_MT19937, 1);  // Determinist mode

  // Read system.dat input file
  readSystemDatFile(&initcf, &nsp, &nitmax, &atoms, &nspps, &natoms, &units,
                    &keyp, &cl, &bl2, &al, &bl, &rc2, &rc, &itp, &rho, &sigma_o,
                    &eps_o);

  if (strcmp(initcf, "dlp") ==
      0) {  // If configuration mode is dlp read it from CONFIG input file
    readConfigFile(&a, &b, &c, nsp, natoms, &ntype, &r, nspps, atoms, &side,
                   &runit, &v0, sigma_o);
  } else if (strcmp(initcf, "lattice") ==
             0) {  // If configuration mode is lattice create a lattice
                   // configuration from scratch
    initLattice(rho, &a, &b, &c, nsp, natoms, &ntype, &r, nspps, atoms, &side,
                &runit, &v0, &stream, sigma_o);
  } else {
    printf("ERROR: '%s' not supported as input configuration!\n", initcf);
    exit(1);
  }
  free(initcf);

  // Read runMC.dat input file
  readRunMCFile(&ensemble, &nstep, &nequil, &nb, &wc, &rdmax, &temp, &deltaeng,
                &ehisto, &vdmax, &scaling, &pres, &deltar, &sideav, &rhisto,
                eps_o, sigma_o, &chpotnb, &chpotit, &final_sm_rate);

  // Initialize output files
  initOutputFiles(ensemble, nsp, atoms);

  // Write initial configuration to conf.xyz output file
  writeConf(natoms, nspps, atoms, r, runit);

  // Initialize simulation box potential
  initPotential(&kt, &pres, temp, units, &rcut, rc, nsp, &rcut2, eps_o);

  // Print simulation's information
  printf("\nSimulating %s ensemble in %s mode.\n", ensemble, runMode);
  printf("No electrostatics.\nEnergy units (%s).\nDensity %lf (npart/A^3).\n",
         units, natoms / (v0 * sigma_o * sigma_o * sigma_o));
  printf("Temperature %.2lf %s\n", temp, units);
  if (strcmp(ensemble, "nvt") == 0 && chpotnb != 0) {
    printf("Chemical potential enabled.\n");
  } else if (strcmp(ensemble, "nvt") == 0 && chpotnb == 0) {
    printf("Chemical potential not enabled.\n");
  }
  putchar('\n');

  // Get initial energy of the configuration
  if (strcmp(runMode, "serial") == 0) {
    esr = energycpu(natoms, itp, r, runit, rc2, nsp, nspps, keyp, al, bl, cl,
                    bl2);
  }
  if (strcmp(runMode, "gpu") == 0) {
    // Initialize GPU (mode 0)
    gpu(0, natoms, itp, r, runit, rc2, nsp, nspps, keyp, al, bl, cl, bl2,
        nitmax, cudadevice, &ntrial, &stream, rdmax, kt, &esr, &naccept,
        chpotit, &v0, side, a, b, c, vdmax, scaling, pres, &nvaccept);
    // Energy GPU (mode 1)
    gpu(1, natoms, itp, r, runit, rc2, nsp, nspps, keyp, al, bl, cl, bl2,
        nitmax, cudadevice, &ntrial, &stream, rdmax, kt, &esr, &naccept,
        chpotit, &v0, side, a, b, c, vdmax, scaling, pres, &nvaccept);
  }

  // Print initial energy of the configuration
  puts(
      "------------------------------------------------------------------------"
      "--------");
  printf("Etotal = %.3lf %s\n", esr * eps_o, units);
  puts(
      "------------------------------------------------------------------------"
      "--------\n");

  // Print output headers
  if (strcmp(ensemble, "nvt") == 0) {
    puts("No. moves  % accept.        E_tot           E_sr");
    puts(
        "----------------------------------------------------------------------"
        "----------");
  } else if (strcmp(ensemble, "npt") == 0) {
    puts(
        " No. moves  % accept.    % acc. vol.    E_tot           E_sr      "
        "Vol");
    puts(
        "----------------------------------------------------------------------"
        "----------");
  }

  // Every parallelized algorithm control time
  double moveAtomsTime = 0.0;
  double moveVolumeTime = 0.0;
  double chPotentialTime = 0.0;

  for (int istep = 1; istep <= nstep; ++istep) {  // For every simulation step

    clock_t mAtomsBegin = clock();  // Move atoms control time begin
    // Move atoms algorithm in selected run mode
    if (strcmp(runMode, "serial") == 0) {
      moveAtoms(&ntrial, natoms, &stream, rdmax, runit, r, nsp, nspps, itp, rc2,
                keyp, al, bl, cl, bl2, kt, &esr, &naccept);
    }
    if (strcmp(runMode, "gpu") == 0) {
      // MoveAtoms GPU (mode 2)
      gpu(2, natoms, itp, r, runit, rc2, nsp, nspps, keyp, al, bl, cl, bl2,
          nitmax, cudadevice, &ntrial, &stream, rdmax, kt, &esr, &naccept,
          chpotit, &v0, side, a, b, c, vdmax, scaling, pres, &nvaccept);
    }
    // Move atoms control time end
    clock_t mAtomsEnd = clock();
    // Time for move atoms algorithm
    moveAtomsTime += (double)(mAtomsEnd - mAtomsBegin) / CLOCKS_PER_SEC;

    if (strcmp(ensemble, "npt") == 0) {  // If simulation ensemble is npt
      clock_t mVolumeBegin = clock();    // Move volume time begin
      // Move volume algorithm in selected run mode
      if (strcmp(runMode, "serial") == 0) {
        moveVolume(&esr, &v0, side, a, b, c, runit, &stream, vdmax, scaling,
                   pres, kt, natoms, &nvaccept, itp, r, rc2, nsp, nspps, keyp,
                   al, bl, cl, bl2);
      }
      if (strcmp(runMode, "gpu") == 0) {
        // MoveVolume GPU (mode 4)
        gpu(4, natoms, itp, r, runit, rc2, nsp, nspps, keyp, al, bl, cl, bl2,
            nitmax, cudadevice, &ntrial, &stream, rdmax, kt, &esr, &naccept,
            chpotit, &v0, side, a, b, c, vdmax, scaling, pres, &nvaccept);
      }
      clock_t mVolumeEnd = clock();  // Move volume control time end
      moveVolumeTime += (double)(mVolumeEnd - mVolumeBegin) /
                        CLOCKS_PER_SEC;  // Time for move volume algorithm
    }

    if (istep >= nequil) {  // If equilibrium phase is over
      histograms(ensemble, &etotal, ehisto, natoms, esr, v0, eref, deltaeng,
                 deltar, rhisto);  // Generate histograms

      if (strcmp(ensemble, "nvt") == 0) {  // If simulation ensemble is nvt
        if (chpotnb != 0 &&
            istep % chpotnb == 0) {  // If chemical potential is enabled and it
                                     // must be calculated
          clock_t chPotentialBegin = clock();  // Chemical potential time begin
          // Chemical potential algorithm in selected run mode
          if (strcmp(runMode, "serial") == 0) {
            chpotentialcpu(chpotit, natoms, itp, r, runit, rc2, nsp, nspps,
                           keyp, al, bl, cl, bl2, kt);
          }
          if (strcmp(runMode, "gpu") == 0) {
            // Chemical Potential GPU (mode 3)
            gpu(3, natoms, itp, r, runit, rc2, nsp, nspps, keyp, al, bl, cl,
                bl2, nitmax, cudadevice, &ntrial, &stream, rdmax, kt, &esr,
                &naccept, chpotit, &v0, side, a, b, c, vdmax, scaling, pres,
                &nvaccept);
          }
          clock_t chPotentialEnd = clock();  // Chemical potential time end
          chPotentialTime +=
              (double)(chPotentialEnd - chPotentialBegin) /
              CLOCKS_PER_SEC;  // Time for chemical potential algorithm
        }
      }
    }

    if (istep % nb == 0) {   // If statistics must be calculated
      if (istep > nequil) {  // If equilibrium phase is over
        // Write statistics in output files
        averages(&firstRun, ensemble, ehisto, deltaeng, &eref, &volav, v0, esr,
                 sideav, side, rhisto, deltar, &naver, &etotal, &etav, &esav,
                 natoms, ntrial, naccept, nvaccept, sigma_o, eps_o,
                 final_sm_rate, &vdmax, rdmax);
      } else {
        // Print results and not write them to output files
        printout(false, &etotal, &eref, esr, ensemble, sideav, etav, naver, v0,
                 esav, volav, side, natoms, ntrial, naccept, nvaccept, sigma_o,
                 eps_o, final_sm_rate, &vdmax, rdmax);
      }
    }

    // If configuration file output's write is enabled and it must be writed
    if (wc != 0 && istep % wc == 0) {
      writeConf(natoms, nspps, atoms, r, runit);
    }
  }

  // Get final energy of the configuration
  if (strcmp(runMode, "serial") == 0) {
    esr = energycpu(natoms, itp, r, runit, rc2, nsp, nspps, keyp, al, bl, cl,
                    bl2);
  }
  if (strcmp(runMode, "gpu") == 0) {
    // Energy GPU (mode 1)
    gpu(1, natoms, itp, r, runit, rc2, nsp, nspps, keyp, al, bl, cl, bl2,
        nitmax, cudadevice, &ntrial, &stream, rdmax, kt, &esr, &naccept,
        chpotit, &v0, side, a, b, c, vdmax, scaling, pres, &nvaccept);
    // Release GPU (mode 5)
    gpu(5, natoms, itp, r, runit, rc2, nsp, nspps, keyp, al, bl, cl, bl2,
        nitmax, cudadevice, &ntrial, &stream, rdmax, kt, &esr, &naccept,
        chpotit, &v0, side, a, b, c, vdmax, scaling, pres, &nvaccept);
  }

  // Print final energy of the configuration
  puts(
      "------------------------------------------------------------------------"
      "--------");
  printf("Etotal = %.3lf %s\n", esr * eps_o, units);
  puts(
      "------------------------------------------------------------------------"
      "--------\n");

  // Write final configuration to conf.xyz output file
  if (wc == 0) writeConf(natoms, nspps, atoms, r, runit);

  clock_t end = clock();  // General time control end
  double time_spent =
      (double)(end - begin) / CLOCKS_PER_SEC;  // Simulation total time

  // Print both simulation total and every concrete algorithm time (in minutes)
  printf("** TCPU/TGPU: %lf **\n", time_spent / 60.0);
  printf("** T_M_Atoms: %lf **\n", moveAtomsTime / 60.0);
  if (strcmp(ensemble, "npt") == 0) {
    printf("** T_M_Volum: %lf **\n", moveVolumeTime / 60.0);
  }
  if (strcmp(ensemble, "nvt") == 0 && chpotnb != 0) {
    printf("** T_Che_Pot: %lf **\n", chPotentialTime / 60.0);
  }

  // Release memory
  vslDeleteStream(&stream);
  for (int i = 0; i < nsp; ++i) {
    free(itp[i]);
    free(rc[i]);
    free(atoms[i]);
  }
  free(keyp);
  free(units);
  free(runMode);
  free(cl);
  free(bl2);
  free(al);
  free(bl);
  free(rc2);
  free(itp);
  free(rc);
  free(atoms);
  free(ntype);
  free(a);
  free(b);
  free(c);
  free(r);
  free(side);
  free(runit);
  free(rdmax);
  free(ensemble);
  free(nspps);
  if (strcmp(ensemble, "nvt") == 0) {
    free(ehisto);
  } else if (strcmp(ensemble, "npt") == 0) {
    free(rhisto);
    free(scaling);
    free(sideav);
  }

  putchar('\n');
  return 0;
}
