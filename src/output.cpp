/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic
 * simulations.
 *
 * Output code file.
 *
 * Author: adpozuelo@gmail.com
 * Version: 1.2.
 * Date: 11/2020
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/io.h"

// Print average and/or instant results and write them to output files
void printout(const bool inst, double *etotal, double *eref, const double esr,
              const char *ensemble, double *sideav, const double etav,
              const unsigned long int naver, const double v0, const double esav,
              const double volav, double *side, const int natoms,
              const unsigned long int ntrial, const unsigned long int naccept,
              const unsigned long int nvaccept, const double sigma_o,
              const double eps_o, const double final_sm_rate, double *vdmax,
              double *rdmax) {
  static double etavq = 0.0;      // total energy average
  static int npeq = 0;            // number of printouts
  unsigned long int ncycles = ntrial / natoms;  // number of cycles
  double n_accept_trial = naccept / (double)ntrial;
  double nv_accept_ncycles = nvaccept / (double)ncycles;
  double nacceptper = 100 * n_accept_trial;  // % of accepted atom's movements
  double nvacceptper =
      100 * nv_accept_ncycles;  // % of accepted volume's movements
  *etotal = esr;

  if (inst) {  // write average information to a output file boolean control
    const char *filenameTh = "../results/thermoaver.dat";  // output file
    FILE *ioth = fopen(filenameTh, "a");  // open file in append mode
    if (ioth == NULL) {
      printf("ERROR: cannot access '%s' file!\n", filenameTh);
      exit(1);
    }

    if (strcmp(ensemble, "nvt") == 0) {  // if ensemble is nvt write information
                                         // (not normalized!) to the output file
      if (fprintf(ioth, "     %2lu    %.4lf        %.3lf       %.3lf\n", ncycles,
                  nacceptper, etav * eps_o / naver,
                  esav * eps_o / naver) == EOF) {
        printf("ERROR: cannot write to '%s' file!\n", filenameTh);
        exit(1);
      }
    } else if (strcmp(ensemble, "npt") ==
               0) {  // if ensemble is npt write information (not normalized!)
                     // to the output file
      if (fprintf(ioth,
                  "     %2lu    %.4lf    %.4lf    %.3lf       %.3lf      %.2lf",
                  ncycles, nacceptper, nvacceptper, etav * eps_o / naver,
                  esav * eps_o / naver,
                  volav * sigma_o * sigma_o * sigma_o / naver) == EOF) {
        printf("ERROR: cannot write to '%s' file!\n", filenameTh);
        exit(1);
      }

      for (int i = 0; i < NDIM; ++i) {  // write volume information (not
                                        // normalized!) to the output file
        if (fprintf(ioth, "      %.5lf", sideav[i] * sigma_o / naver) == EOF) {
          printf("ERROR: cannot write to '%s' file!\n", filenameTh);
          exit(1);
        }
      }

      if (fputc('\n', ioth) == EOF) {
        printf("ERROR: cannot write to '%s' file!\n", filenameTh);
        exit(1);
      }
    }

    fclose(ioth);  // close output file

  } else {
    for (int i = 0; i < NDIM; ++i) {
      rdmax[i] *= (n_accept_trial / final_sm_rate);
    }
    if (strcmp(ensemble, "npt") == 0) {
      *vdmax *= (nv_accept_ncycles / final_sm_rate);
    }

    npeq++;                // update printout counter
    etavq += *etotal;      // update total energy average
    *eref = etavq / npeq;  // set eref energy
  }

  // Always print instant information and write it to the output file
  const char *filenameThi = "../results/thermoins.dat";  // output filename
  FILE *iothi = fopen(filenameThi, "a");  // write file in append mode
  if (iothi == NULL) {
    printf("ERROR: cannot access '%s' file!\n", filenameThi);
    exit(1);
  }

  if (strcmp(ensemble, "nvt") == 0) {  // if ensemble is nvt write information
                                       // (not normalized!) to the output file
    if (fprintf(iothi, "     %2lu    %.4lf        %.3lf       %.3lf\n", ncycles,
                nacceptper, *etotal * eps_o, esr * eps_o) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filenameThi);
      exit(1);
    }
    // and print information (not normalized!)
    printf("     %2lu    %.4lf        %.3lf       %.3lf\n", ncycles, nacceptper,
           *etotal * eps_o, esr * eps_o);
  } else if (strcmp(ensemble, "npt") ==
             0) {  // if ensemble is npt write information (not normalized!) to
                   // the output file
    if (fprintf(
            iothi,
            "     %2lu    %.4lf        %.4lf         %.3lf      %.3lf     %.2lf",
            ncycles, nacceptper, nvacceptper, *etotal * eps_o, esr * eps_o,
            v0 * sigma_o * sigma_o * sigma_o) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filenameThi);
      exit(1);
    }

    for (int i = 0; i < NDIM; ++i) {  // write volume information (not
                                      // normalized!) to the output file
      if (fprintf(iothi, "  %.5lf", side[i] * sigma_o) == EOF) {
        printf("ERROR: cannot write to '%s' file!\n", filenameThi);
        exit(1);
      }
    }

    if (fputc('\n', iothi) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filenameThi);
      exit(1);
    }
    // and print information (not normalized!)
    printf("     %2lu     %.4lf      %.4lf   %.3lf     %.3lf   %.2lf\n", ncycles,
           nacceptper, nvacceptper, *etotal * eps_o, esr * eps_o, v0 * sigma_o);
  }

  fclose(iothi);  // close output file
}

// Initialize output files
void initOutputFiles(const char *ensemble, const int nsp, char **atoms) {
  const char *filenameConf = "../results/conf.xyz";  // output filename
  FILE *fp;
  fp = fopen(filenameConf, "w");  // open file in write mode
  fclose(fp);                     // close output file

  const char *filenameChPot = "../results/chpotential.dat";  // output filename
  FILE *fpChPot;
  fpChPot = fopen(filenameChPot, "w");  // open file in write mode
  for (int i = 0; i < nsp; ++i) {
    // write to output file the specie's chemical character representation as a
    // header
    if (fprintf(fpChPot, "%s\t\t", atoms[i]) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filenameChPot);
      exit(1);
    }
  }
  if (fputc('\n', fpChPot) == EOF) {
    printf("ERROR: cannot write to '%s' file!\n", filenameChPot);
    exit(1);
  }
  fclose(fpChPot);  // close output file

  const char *filenameTh = "../results/thermoaver.dat";  // output filename
  const char *filenameThi = "../results/thermoins.dat";  // output filename
  FILE *ioth = fopen(filenameTh, "w");  // open file in write mode
  if (ioth == NULL) {
    printf("ERROR: cannot access '%s' file!\n", filenameTh);
    exit(1);
  }
  FILE *iothi = fopen(filenameThi, "w");  // open file in write mode
  if (ioth == NULL) {
    printf("ERROR: cannot access '%s' file!\n", filenameThi);
    exit(1);
  }

  if (strcmp(ensemble, "nvt") ==
      0) {  // if ensemble is nvt write the header to output file
    if (fputs("No. moves  % accept.       <E_tot>         <E_sr>\n", ioth) ==
        EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filenameTh);
      exit(1);
    }

    for (int i = 0; i < 62; ++i) {
      if (fputc('-', ioth) == EOF) {
        printf("ERROR: cannot write to '%s' file!\n", filenameTh);
        exit(1);
      }
    }

    if (fputc('\n', ioth) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filenameTh);
      exit(1);
    }
    // if ensemble is nvt write the header to output file
    if (fputs("No. moves  % accept.        E_tot           E_sr\n", iothi) ==
        EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filenameThi);
      exit(1);
    }

    for (int i = 0; i < 62; ++i) {
      if (fputc('-', iothi) == EOF) {
        printf("ERROR: cannot write to '%s' file!\n", filenameThi);
        exit(1);
      }
    }

    if (fputc('\n', iothi) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filenameThi);
      exit(1);
    }

  } else if (strcmp(ensemble, "npt") ==
             0) {  // if ensemble is npt write the header to output file
    if (fputs("No. moves  % accept.   % accept. vol.   <E_tot>      <E_sr>    "
              "<Vol>      <Lx>     <Ly>    <Lz>\n",
              ioth) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filenameTh);
      exit(1);
    }

    for (int i = 0; i < 100; ++i) {
      if (fputc('-', ioth) == EOF) {
        printf("ERROR: cannot write to '%s' file!\n", filenameTh);
        exit(1);
      }
    }

    if (fputc('\n', ioth) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filenameTh);
      exit(1);
    }
    // if ensemble is nvt write the header to output file
    if (fputs("No. moves  % accept.   % acc. vol.    E_tot          E_sr       "
              "   Vol       Lx        Ly        Lz\n",
              iothi) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filenameThi);
      exit(1);
    }

    for (int i = 0; i < 100; ++i) {
      if (fputc('-', iothi) == EOF) {
        printf("ERROR: cannot write to '%s' file!\n", filenameThi);
        exit(1);
      }
    }

    if (fputc('\n', iothi) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filenameThi);
      exit(1);
    }
  }

  // close output files
  fclose(ioth);
  fclose(iothi);
}

// Write configuration to conf.xyz output file
void writeConf(const int natoms, int *nspps, char **atoms, double *r,
               double *runit) {
  double *rr = (double *)malloc(natoms * NDIM * sizeof(double));
  for (int i = 0; i < natoms; ++i) {
    for (int j = 0; j < NDIM; ++j) {
      rr[i * NDIM + j] = r[i * NDIM + j] *
                         runit[j];  // not normalized particles positions (xyz)
    }
  }

  const char *filename = "../results/conf.xyz";  // output filename

  FILE *fp;
  fp = fopen(filename, "a");  // open file in append mode
  if (fprintf(fp, "%d\n Initial Configuration\n", natoms) ==
      EOF) {  // write header to ouput file
    printf("ERROR: cannot write to '%s' file!\n", filename);
    exit(1);
  }

  int nsppsc = 0;
  for (int i = 0; i < natoms; ++i) {
    if (i < nspps[nsppsc]) {
      if (fprintf(fp, "%s ", atoms[nsppsc]) ==
          EOF) {  // write specie's chemical character representation to output
                  // file
        printf("ERROR: cannot write to '%s' file!\n", filename);
        exit(1);
      }
    } else {
      ++nsppsc;
      if (fprintf(fp, "%s ", atoms[nsppsc]) == EOF) {
        printf("ERROR: cannot write to '%s' file!\n", filename);
        exit(1);
      }
    }
    for (int j = 0; j < NDIM; ++j) {
      if (fprintf(fp, "%lf ", rr[i * NDIM + j]) ==
          EOF) {  // write particle positions (xyz) to output file
        printf("ERROR: cannot write to '%s' file!\n", filename);
        exit(1);
      }
    }
    if (fputc('\n', fp) == EOF) {
      printf("ERROR: cannot write to '%s' file!\n", filename);
      exit(1);
    }
  }
  fclose(fp);  // close output file
  free(rr);    // release memory
}
