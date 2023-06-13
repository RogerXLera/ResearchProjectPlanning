#!/bin/bash

function wait_empty_queue {
    while true
    do
        if hash condor_submit 2>/dev/null
        then
            remaining=$( condor_q | tail -n 3 | head -n 1 | cut -f 4 -d\  )
        elif hash sbatch 2>/dev/null
        then
            remaining=$( squeue --me -h | wc -l )
        else
            echo "Unknown cluster"
            exit 1
        fi
        echo "$remaining jobs remaining" 
        if [[ $remaining == 0 ]]
        then
            return
        else
            echo "Waiting queue to be empty..." 
            sleep 60
        fi
    done
}


start=0
n=1348 #number of jobs
m=10 #number of periods
d=20 #dedication per periods
alpha=1.0 
beta=0.0
delta=0.0
step=200

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
            start="$1"
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
            delta="$1"
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
for j in $( seq $start $n )
do
    if [[ $i == $step ]]
    then
        wait_empty_queue
        i=0
    fi
    echo "Submitted DPP ($j)"
    bash submit.sh -m $m -d $d -j $j --alpha $alpha --beta $beta --delta $delta $args
    i=$((i+1))

done