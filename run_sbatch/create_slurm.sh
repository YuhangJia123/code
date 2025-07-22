#!/bin/bash

peram_dir="/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light"
#peram_dir="/beegfs/group/lqcd/eigensystem/beta6.41_mu-0.2295_ms-0.2050_L32x96"
for conf in {4050..4050..50}
# for conf in {4050..26700..50}
do
while (( $(squeue -p gpu-debug  -u jiayu | wc -l) > 4 ))
do
    sleep 10
done
if [ -d $peram_dir/${conf} ]; then
sed "s/=CONF=/$conf/g" submit_slurm.sh > submit.$conf.sh
chmod +x submit.$conf.sh
sbatch submit.$conf.sh
fi
done
