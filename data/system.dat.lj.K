dlp ## initial configuration
2 ## number of species (sp)
O 3072 ## name of sp1, number of particles of sp1
Si 1536 ## name of spN, number of particles of spN
K ## energy units
lj                  !! type of 1-1 interaction
120.0     2.791  3.0  !! O-O  a b  rc (LJ parameters + cutoff) ordered as upper triangular matrix
lj                  !! type of 1-2 interaction
100.000   1.628  3.0  !! O-Si
lj                  !! type of 1-3 interaction
85.0  2.1598 3.0  !! Si-Si

