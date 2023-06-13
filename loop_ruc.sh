#!/bin/bash

alpha=0.8
alpha_list=$(0.25 0.5 0.75 1.0)
beta=0.00001
beta_list=$(0.0 0.000001 0.00001 0.0001 0.001)
gamma=0.00001
gamma_list=$(0.0 0.00001 1000000)
mu=1.0
mu_list=$(0.95 1.0 1.05)
n=10
nproj=3
prob=0.1
w_delta=0.1
w_rho=0.1
nseed=10

while [[ $# > 0 ]]
do
    key="$1"
    case $key in
        -n)
            shift
            n="$1"
            shift
        ;;
        --start)
            shift
            nseed="$1"
            shift
        ;;
        --alpha)
            shift
            alpha="$1"
            shift
        ;;
        --beta)
            shift
            beta="$1"
            shift
        ;;
        --delta)
            shift
            gamma="$1"
            shift
        ;;
        -m)
            shift
            m="$1"
            shift
        ;;
        -d)
            shift
            d="$1"
            shift
        ;;
        --step)
            shift
            step="$1"
            shift
        ;;
        *)
            args="$args$key "
            shift
        ;;
    esac
done

i=0
for seed in $( seq 0 $nseed )
do
    python3 solve.py --seed $seed -a $alpha -b $beta -g $gamma -m $mu -I $n --nproj $nproj -p $prob --delta $w_delta --rho $w_rho --robust
    echo python3.7 solve.py --seed $seed -a $alpha -b $beta -g $gamma -m $mu -n $n --nproj $nproj -p $prob --delta $w_delta --rho $w_rho --robust
done