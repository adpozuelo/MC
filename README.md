Serial and parallel (CUDA) general purpose Monte Carlo code for atomistic simulations.
===========

<p> Developer:
	Antonio Díaz Pozuelo (adpozuelo@gmail.com) </p>

<p> MC is my URV's (Universitat Rovira i Virgili) / UOC's (Universitat Oberta de Catalunya) final master project (TFM). </p>

<p> MC is a Metropolis Monte Carlo which simulates bulk systems composed of soft spherical particles without charges. 
MC implements ensembles NVT and NPT.
Chemical potential is implemented only for NVT ensemble.</p>

There are examples of input data files with atom's systems (dlp) in "data" directory.

Design & Arquitecture
==========

MC is designed to use the massive parallel paradigm for intensive computation like Metropolis move atoms, move volume and chemical potential algorithms. Thus, MC needs a NVIDIA's VGA with CUDA arquitecture which must support compute capability 3.0 or higher.

Implementation
==========
MC is implemented with NVIDIA CUDA SDK 8.0 and it uses the Intel C/C++ compiler.

Requisites
==========

- Software:

  * NVIDIA CUDA Compiler (nvcc) with cuRand support.
  * Intel C/C++ Compiler (icpc) with MKL (Math Kernel Library) support.

- Hardware:

  * NVIDIA's VGA CUDA capable arquitecture which must support compute capability 3.0 or higher.

Install
=======

<p> Download MC application: </p>

	git clone https://github.com/adpozuelo/MC.git
	cd MC

<p> Compile </b>: </p>

	make

<p> Execute MC application: </p>

	cd bin
	./mc.exe serial  <- execute MC in serial mode.
	./mc.exe gpu 0   <- execute MC in parallel mode using cuda device 0.
	./mc.exe 	 <- for help.
		
