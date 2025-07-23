#!/bin/bash
peram_dir="/home/HDD/light/light/"
P_max=1
for conf in {2700..2700..20}
#for conf in {4050..26700..50}
do
    for Px in {0..1..1}
    do
        for Py in {0..1..1}
        do
            for Pz in {0..1..1}
            do
                if [ -d ${peram_dir}/${conf} ]; then
                chmod +x ./input.sh
                ./input.sh $conf $Px $Py $Pz > ./run_created/input/input_${conf}_${Px}_${Py}_${Pz}
                fi
            done
        done
    done
done

