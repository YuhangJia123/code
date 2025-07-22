#!/bin/bash

peram_dir="/public/group/lqcd/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light"
for conf in {4050..4050..50}
#for conf in {4050..26700..50}
do
if [ -d ${peram_dir}/${conf} ]; then
./input.sh $conf > input_${conf}
fi
done
