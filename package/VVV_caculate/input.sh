#!/bin/bash
cat << EOF
Nt 96
Nx 48
conf_id $1
Nev 100
Nev1 100
nMom 1
nproc 8
Px $2
Py $3
Pz $4
peram_u_dir /home/HDD/light/light/$1
peram_s_dir /home/HDD/strange/strange/$1
peram_c_dir /home/HDD/charm/charm/$1
corr_dir ../corr
eigen_dir /home/HDD/eigen/eigensystem/$1
VVV_save_dir /home/jiayuhang/research_wu/projects/K+N/code_debug/test_data/VVV.Px$2Py$3Pz$4.conf$1
EOF
