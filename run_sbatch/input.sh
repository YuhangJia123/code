#!/bin/bash
cat << EOF
Nt 72
Nx 24
conf_id $1
Nev 100
Nev1 100
nMom 1
nproc 8
peram_u_dir /public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/$1
peram_s_dir /public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/strange/$1
peram_c_dir /public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/charm/$1
corr_pion_dir /public/home/jiayu/research/Projects/test_by_wangzl/source_code_for_K+N/P001/corr
corr_kaon_dir /public/home/jiayu/research/Projects/test_by_wangzl/source_code_for_K+N/P001/corr
corr_etas_dir /public/home/jiayu/research/Projects/test_by_wangzl/source_code_for_K+N/P001/corr
corr_dir /public/home/jiayu/research/Projects/test_by_wangzl/source_code_for_K+N/P002_G1-/corr_nobase
eigen_dir /public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/$1
phi_dir /public/group/lqcd/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/$1/VVV
all_operator_dir /public/home/jiayu/research/Projects/test_by_wangzl/source_code_for_K+N/source_code/O_group/nobase_output/operators.npz
EOF
