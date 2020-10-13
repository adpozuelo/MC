/*
 * TFM (URV/UOC): Computational Engineering and Mathematics.
 * Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic
 * simulations.
 *
 * Input code file.
 *
 * Author: adpozuelo@gmail.com
 * Version: 2.0
 * Date: 2020
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/io.h"

// Read a line from a file
void readLine(FILE *file, const char *filename, char *line) {
  if (fgets(line, MAX_LINE_SIZE, file) == NULL) {
    printf("ERROR: Abrupt '%s' End-Of-File reached!\n", filename);
    exit(1);
  }
}

// Read system.dat input file
void readSystemDatFile(char **initcf, int *nsp, int *nitmax, char ***atoms,
                       int **nspps, int *natoms, char **units, int **keyp,
                       double **cl, double **bl2, double **al, double **bl,
                       double **rc2, double ***rc, int ***itp, double *rho,
                       double *sigma_o, double *eps_o) {
  char line[MAX_LINE_SIZE] = "";
  char buffer[MAX_LINE_SIZE] = "";
  const char *filename = "../data/system.dat";  // input filename
  const char *errorIF = "ERROR: Invalid '../data/system.dat' file format!\n";
  FILE *systemDatFile = fopen(filename, "r");  // open file in read-only mode
  if (systemDatFile == NULL) {
    printf("ERROR: cannot access '%s' file!\n", filename);
    exit(1);
  }
  readLine(systemDatFile, filename, line);
  if (sscanf(line, "%s", buffer) < 1) {
    fputs(errorIF, stderr);
    exit(1);
  }
  *initcf = (char *)malloc(strlen(buffer) * sizeof(char));
  if (*initcf == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  strcpy(*initcf, buffer);  // read the configuration model (dlp or lattice)
  readLine(systemDatFile, filename, line);
  if (strcmp(*initcf, "dlp") == 0) {    // if configuration model is dlp
    if (sscanf(line, "%d", nsp) < 1) {  // read the number os species
      fputs(errorIF, stderr);
      exit(1);
    }
    // if configuration model is lattice
  } else if (strcmp(*initcf, "lattice") == 0) {
    // read both the configuration model and the simulation box density
    if (sscanf(line, "%d %lf", nsp, rho) < 1) {
      fputs(errorIF, stderr);
      exit(1);
    }
  } else {
    printf("ERROR: '%s' not supported as input configuration!\n", *initcf);
    exit(1);
  }
  // calculate the max number of interactions between species
  *nitmax = (*nsp * *nsp + *nsp) / 2;
  *nspps = (int *)malloc(*nsp * sizeof(int));
  *atoms = (char **)malloc(*nsp * sizeof(char *));
  if (*atoms == NULL || *nspps == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  for (int i = 0; i < *nsp; ++i) {  // for every specie
    readLine(systemDatFile, filename, line);
    // read specie's chemical character representation and the number of
    // particles per specie
    if (sscanf(line, "%s %d", buffer, &(*nspps)[i]) < 2) {
      fputs(errorIF, stderr);
      exit(1);
    }
    (*atoms)[i] = (char *)malloc(strlen(buffer) * sizeof(char));
    if ((*atoms)[i] == NULL) {
      fputs(errorNEM, stderr);
      exit(1);
    }
    strcpy((*atoms)[i], buffer);
  }
  for (int i = 0; i < *nsp; ++i) {
    *natoms += (*nspps)[i];  // calculate total number of atoms/particles
  }
  readLine(systemDatFile, filename, line);
  if (sscanf(line, "%s", buffer) < 1) {
    fputs(errorIF, stderr);
    exit(1);
  }
  *units = (char *)malloc(strlen(buffer) * sizeof(char));
  if (*units == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  strcpy(*units, buffer);  // read potential units
  if (strcmp(*units, "eV") != 0 && strcmp(*units, "K") != 0) {
    printf("ERROR: '%s' not implemented as energy unit!\n", *units);
    exit(1);
  }
  double **aa = (double **)malloc(*nsp * sizeof(double *));
  double **bb = (double **)malloc(*nsp * sizeof(double *));
  double **cc = (double **)malloc(*nsp * sizeof(double *));
  *keyp = (int *)malloc(*nitmax * sizeof(int));
  *cl = (double *)malloc(*nitmax * sizeof(double));
  *bl2 = (double *)malloc(*nitmax * sizeof(double));
  *al = (double *)malloc(*nitmax * sizeof(double));
  *bl = (double *)malloc(*nitmax * sizeof(double));
  *rc2 = (double *)malloc(*nitmax * sizeof(double));
  *rc = (double **)malloc(*nsp * sizeof(double *));
  *itp = (int **)malloc(*nsp * sizeof(int *));
  if (aa == NULL || bb == NULL || cc == NULL || *keyp == NULL || *cl == NULL ||
      *bl2 == NULL || *al == NULL || *bl == NULL || *rc2 == NULL ||
      *rc == NULL || *itp == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  for (int i = 0; i < *nsp; ++i) {
    aa[i] = (double *)malloc(*nsp * sizeof(double));
    bb[i] = (double *)malloc(*nsp * sizeof(double));
    cc[i] = (double *)calloc(*nsp, sizeof(double));
    (*rc)[i] = (double *)malloc(*nsp * sizeof(double));
    (*itp)[i] = (int *)malloc(*nsp * sizeof(int));
    if (aa[i] == NULL || bb[i] == NULL || cc[i] == NULL || (*rc)[i] == NULL ||
        (*itp)[i] == NULL) {
      fputs(errorNEM, stderr);
      exit(1);
    }
  }
  int nit = 0;
  for (int i = 0; i < *nsp; ++i) {
    for (int j = i; j < *nsp; ++j) {  // for every interaction between species
      readLine(systemDatFile, filename, line);
      if (sscanf(line, "%s", buffer) < 1) {  // read the potential interaction
        fputs(errorIF, stderr);
        exit(1);
      }

      if (strcmp(buffer, "mors") == 0) {  // if potential is Morse
        (*keyp)[nit] = 1;                 // potential's key -> 1 = Morse
        readLine(systemDatFile, filename, line);
        // read Morse parameters
        if (sscanf(line, "%lf %lf %lf %lf", &aa[i][j], &bb[i][j], &cc[i][j],
                   &(*rc)[i][j]) < 4) {
          fputs(errorIF, stderr);
          exit(1);
        }
        (*cl)[nit] = cc[i][j];  // linearize potential interactions parameters
      } else if (strcmp(buffer, "lj") == 0) {  // if potential is Lennard Jones
        (*keyp)[nit] = 2;  // potential's key -> 2 = Lennard Jones
        readLine(systemDatFile, filename, line);
        // read Lennard Jones parameters
        if (sscanf(line, "%lf %lf %lf", &aa[i][j], &bb[i][j], &(*rc)[i][j]) <
            3) {
          fputs(errorIF, stderr);
          exit(1);
        }
      } else {
        printf("ERROR: Input failed: %s not implemented yet!", buffer);
        exit(1);
      }
      // linearize potential interactions parameters
      (*al)[nit] = aa[i][j];
      (*bl)[nit] = bb[i][j];
      (*rc)[j][i] = (*rc)[i][j];
      (*rc2)[nit] = (*rc)[i][j] * (*rc)[i][j];  // power 2 cutoff parameter
      (*itp)[i][j] = nit;
      (*itp)[j][i] = nit;
      ++nit;
    }
  }
  // release memory
  for (int i = 0; i < *nsp; ++i) {
    free(aa[i]);
    free(bb[i]);
    free(cc[i]);
  }
  free(aa);
  free(bb);
  free(cc);
  fclose(systemDatFile);  // close file
  // normalize interaction potential parameters for optimize floating point
  // representation errors
  if ((*keyp)[0] == 1) {
    *sigma_o = (*cl)[0];
    for (int i = 0; i < *nitmax; ++i) {
      (*cl)[i] /= *sigma_o;
      (*bl)[i] *= *sigma_o;
    }
  } else if ((*keyp)[0] == 2) {
    *sigma_o = (*bl)[0];
    for (int i = 0; i < *nitmax; ++i) {
      (*bl)[i] /= *sigma_o;
      (*bl2)[i] =
          (*bl)[i] * (*bl)[i];  // power 2 potential interactions parameter
    }
  } else {
    fputs("ERROR: interaction not implemented!\n", stderr);
    exit(1);
  }
  for (int i = 0; i < *nsp; ++i) {
    for (int j = 0; j < *nsp; ++j) {
      (*rc)[i][j] /= *sigma_o;
    }
  }
  nit = 0;
  for (int i = 0; i < *nsp; ++i) {
    for (int j = i; j < *nsp; ++j) {
      (*rc2)[nit] = (*rc)[i][j] * (*rc)[i][j];
      ++nit;
    }
  }
  *eps_o = (*al)[0];
  for (int i = 0; i < *nitmax; ++i) {
    (*al)[i] /= *eps_o;
  }
}

// Read CONFIG input file
void readConfigFile(double **a, double **b, double **c, const int nsp,
                    const int natoms, int **ntype, double **r, int *nspps,
                    char **atoms, double **side, double **runit, double *v0,
                    const double sigma_o) {
  char line[MAX_LINE_SIZE] = "";
  char buffer[MAX_LINE_SIZE] = "";
  const char *filename = "../data/CONFIG";  // input filename
  const char *errorInvalidFormatConfigFile =
      "ERROR: Invalid '../data/CONFIG' file format!\n";
  FILE *configFile = fopen(filename, "r");  // open file in read-only mode
  if (configFile == NULL) {
    printf("ERROR: cannot access '%s' file!\n", filename);
    exit(1);
  }
  readLine(configFile, filename, line);
  int keytrj;
  readLine(configFile, filename, line);
  // read trajectories description dimensions
  if (sscanf(line, "%d", &keytrj) < 1) {
    fputs(errorInvalidFormatConfigFile, stderr);
    exit(1);
  }
  *a = (double *)calloc(NDIM, sizeof(double));
  *b = (double *)calloc(NDIM, sizeof(double));
  *c = (double *)calloc(NDIM, sizeof(double));
  if (*a == NULL || *b == NULL || *c == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  for (int i = 0; i < NDIM; ++i) {
    readLine(configFile, filename, line);
    // read the simulations's box dimensions (XYZ)
    if (sscanf(line, "%lf %lf %lf", &(*a)[i], &(*b)[i], &(*c)[i]) < 3) {
      fputs(errorInvalidFormatConfigFile, stderr);
      exit(1);
    }
  }
  // normalize the simulation's box dimensions in reference to potential
  // parameters
  for (int i = 0; i < NDIM; ++i) {
    (*a)[i] /= sigma_o;
    (*b)[i] /= sigma_o;
    (*c)[i] /= sigma_o;
  }
  // Check ortho-rombic condition
  if ((*a)[1] * (*a)[1] + (*a)[2] * (*a)[2] + (*b)[0] * (*b)[0] +
          (*b)[2] * (*b)[2] + (*c)[0] * (*c)[0] + (*c)[1] * (*c)[1] >
      1.0e-7) {
    fputs("ERROR: Non orthorombic cells not implemented!\n", stderr);
    exit(1);
  }
  *ntype = (int *)calloc(nsp, sizeof(int));
  *r = (double *)malloc((natoms)*NDIM * sizeof(double));
  // particles positions (xyz) by specie used by short particles while they are
  // inserted
  double **rnsp = (double **)malloc(nsp * sizeof(double *));
  if (*ntype == NULL || *r == NULL || rnsp == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  for (int i = 0; i < nsp; ++i) {
    rnsp[i] = (double *)malloc(nspps[i] * NDIM * sizeof(double));
    if (rnsp[i] == NULL) {
      fputs(errorNEM, stderr);
      exit(1);
    }
  }
  int iatm;
  for (int i = 0; i < natoms; ++i) {  // for every particle
    readLine(configFile, filename, line);
    // read chemical character representation and the id of the particle
    if (sscanf(line, "%s %d", buffer, &iatm) < 2) {
      fputs(errorInvalidFormatConfigFile, stderr);
      exit(1);
    }
    if (iatm - 1 != i) {  // check particle's id integrity
      fputs(errorInvalidFormatConfigFile, stderr);
      exit(1);
    }
    // for every specie
    for (int j = 0; j < nsp; ++j) {
      // select which specie's vector must be selected
      if (strcmp(buffer, atoms[j]) == 0) {
        readLine(configFile, filename, line);
        // insertion short particle in species vector
        if (sscanf(line, "%lf %lf %lf", &rnsp[j][(*ntype)[j] * NDIM],
                   &rnsp[j][(*ntype)[j] * NDIM + 1],
                   &rnsp[j][(*ntype)[j] * NDIM + 2]) < 3) {
          fputs(errorInvalidFormatConfigFile, stderr);
          exit(1);
        }
        for (int k = 0; k < keytrj; ++k) {  // avoid trajectorie's data
          readLine(configFile, filename, line);
        }
        (*ntype)[j]++;  // count the number of particles of every especie
      }
    }
  }
  fclose(configFile);  // close file
  for (int i = 0; i < nsp; ++i) {
    if ((*ntype)[i] != nspps[i]) {  // integrity check
      fputs("ERROR: ntype does not match!!\n", stderr);
      exit(1);
    }
  }
  for (int i = 1; i < nsp; ++i) {
    for (int j = 0; j < i; ++j) {
      // create the accumulated number of species per specie
      nspps[i] += nspps[j];
    }
  }
  int counter = 0;
  for (int i = 0; i < nsp; ++i) {
    for (int j = 0; j < (*ntype)[i] * NDIM; ++j) {
      // linearize and normalize the shorted by specie particle's positions
      // (XYZ) in reference to potential parameters
      (*r)[counter] = rnsp[i][j] / sigma_o;
      counter++;
    }
  }
  // release memory
  for (int i = 0; i < nsp; ++i) {
    free(rnsp[i]);
  }
  free(rnsp);
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
  for (int i = 0; i < NDIM; ++i) {
    (*side)[i] = (*runit)[i];
  }
  // calculate the volume of the simulation box
  *v0 = (*a)[0] * (*b)[1] * (*c)[2] + (*a)[1] * (*b)[2] * (*c)[0] +
        (*a)[2] * (*b)[0] * (*c)[1] - (*a)[2] * (*b)[1] * (*c)[0] -
        (*a)[1] * (*b)[0] * (*c)[2] - (*a)[0] * (*b)[2] * (*c)[1];
  for (int i = 0; i < natoms; ++i) {
    for (int j = 0; j < NDIM; ++j) {
      // normalize particle's positions (XYZ) in reference to units of the
      // simulation box's length (XYZ) side
      (*r)[i * NDIM + j] /= (*runit)[j];
    }
  }
}

// Read runMC.dat input file
void readRunMCFile(char **ensemble, int *nstep, int *nequil, int *nb, int *wc,
                   double **rdmax, double *temp, double *deltaeng,
                   unsigned long long int **ehisto, double *vdmax,
                   char **scaling, double *pres, double *deltar,
                   double **sideav, unsigned long long int **rhisto,
                   const double eps_o, const double sigma_o, int *chpotnb,
                   int *chpotit, double *final_sm_rate) {
  char line[MAX_LINE_SIZE] = "";
  char buffer[MAX_LINE_SIZE] = "";
  const char *filename = "../data/runMC.dat";  // input filename
  const char *errorInvalidFormatRunFile =
      "ERROR: Invalid '../data/runMC.dat' file format!\n";
  FILE *runFile = fopen(filename, "r");  // open file in read-only mode
  if (runFile == NULL) {
    printf("ERROR: cannot access '%s' file!\n", filename);
    exit(1);
  }
  readLine(runFile, filename, line);
  if (sscanf(line, "%s", buffer) < 1) {
    fputs(errorInvalidFormatRunFile, stderr);
    exit(1);
  }
  *ensemble = (char *)malloc(strlen(buffer) * sizeof(char));
  if (*ensemble == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  strcpy(*ensemble, buffer);  // read simulation ensemble (nvt or npt)
  if (strcmp(*ensemble, "nvt") != 0 && strcmp(*ensemble, "npt") != 0) {
    printf("ERROR: '%s', not implemented yet!\n", *ensemble);
    exit(1);
  }
  readLine(runFile, filename, line);
  // read number of total simulation's steps, number of equilibrium phase steps,
  // every steps averages will be executed and every steps configuration file
  // will be writed variables
  if (sscanf(line, "%d %d %d %d", nstep, nequil, nb, wc) < 4) {
    fputs(errorInvalidFormatRunFile, stderr);
    exit(1);
  }
  if (strcmp(*ensemble, "nvt") == 0) {  // if ensemble is nvt
    readLine(runFile, filename, line);
    // read every steps chemical potential will be executed and chemical
    // potential iterations (number of particles inserted for every specie)
    if (sscanf(line, "%d %d", chpotnb, chpotit) < 2) {
      fputs(errorInvalidFormatRunFile, stderr);
      exit(1);
    }
  }
  *rdmax = (double *)malloc(NDIM * sizeof(double));
  if (*rdmax == NULL) {
    fputs(errorNEM, stderr);
    exit(1);
  }
  if (strcmp(*ensemble, "nvt") == 0) {  // if ensemble is nvt
    readLine(runFile, filename, line);
    // read maximum trial displacements
    if (sscanf(line, "%lf %lf %lf %lf", final_sm_rate, &(*rdmax)[0],
               &(*rdmax)[1], &(*rdmax)[2]) < 4) {
      fputs(errorInvalidFormatRunFile, stderr);
      exit(1);
    }
    readLine(runFile, filename, line);
    if (sscanf(line, "%lf", temp) < 1) {  // read simulation box temperature
      fputs(errorInvalidFormatRunFile, stderr);
      exit(1);
    }
    readLine(runFile, filename, line);
    // read nvt delta grid for energy histogram
    if (sscanf(line, "%lf", deltaeng) < 1) {
      fputs(errorInvalidFormatRunFile, stderr);
      exit(1);
    }
    // create energy histogram
    *ehisto = (unsigned long long int *)calloc(NDE * 2 + 1,
                                               sizeof(unsigned long long int));
    if (*ehisto == NULL) {
      fputs(errorNEM, stderr);
      exit(1);
    }
  }
  if (strcmp(*ensemble, "npt") == 0) {  // if ensemble is ntp
    readLine(runFile, filename, line);
    // read maximum trial displacements, volume maximum displacement and
    // simulation scaling (ortho or isotr)
    if (sscanf(line, "%lf %lf %lf %lf %lf %s", final_sm_rate, &(*rdmax)[0],
               &(*rdmax)[1], &(*rdmax)[2], vdmax, buffer) < 6) {
      fputs(errorInvalidFormatRunFile, stderr);
      exit(1);
    }
    *scaling = (char *)malloc(strlen(buffer) * sizeof(char));
    if (*scaling == NULL) {
      fputs(errorNEM, stderr);
      exit(1);
    }
    strcpy(*scaling, buffer);
    if (strcmp(*scaling, "ortho") != 0 && strcmp(*scaling, "isotr") != 0) {
      printf("ERROR: '%s', not implemented yet!\n", *scaling);
      exit(1);
    }
    readLine(runFile, filename, line);
    // read simulation box temperature and simulation box pression
    if (sscanf(line, "%lf %lf", temp, pres) < 2) {
      fputs(errorInvalidFormatRunFile, stderr);
      exit(1);
    }
    readLine(runFile, filename, line);
    // read npt delta grid for density histogram
    if (sscanf(line, "%lf", deltar) < 1) {
      fputs(errorInvalidFormatRunFile, stderr);
      exit(1);
    }
    *deltar /= sigma_o;  // normalize in reference to potential parameters
    *vdmax /= sigma_o;   // normalize in reference to potential parameters
    *sideav = (double *)calloc(NDIM, sizeof(double));
    // create density histogram
    *rhisto = (unsigned long long int *)calloc(NRMAX + 1,
                                               sizeof(unsigned long long int));
    if (*rhisto == NULL) {
      fputs(errorNEM, stderr);
      exit(1);
    }
  }
  fclose(runFile);  // close file
  for (int i = 0; i < NDIM; ++i) {
    (*rdmax)[i] /= sigma_o;  // normalize in reference to potential parameters
  }
}
