#!/bin/bash

function modify_file() {
    
    local file="$1"  # The file path is passed as an argument
    local m="$2"
    local d="$3"
    local j="$4"

    # Check if the file exists
    if [ ! -f "$file" ]; then
        echo "File '$file' does not exist."
        return 1
    fi

    # Read the file contents into an array
    mapfile -t lines < "$file"

    # Modify the contents (e.g., add a prefix to each line)
    modified_lines=()
    for line in "${lines[@]}"; do
        if [[ $line == job,* ]]; then
            modified_lines+=("job,$j")
        elif [[ $line == dedication,* ]]; then
            modified_lines+=("dedication,$d")
        elif [[ $line == nperiods,* ]]; then
            modified_lines+=("nperiods,$m")
        else
            modified_lines+=("$line")
        fi
    done

    # Write the modified contents back to the file
    printf '%s\n' "${modified_lines[@]}" > "$file"

    echo "File '$file' modified successfully."
}


m=10 #number of periods
d=20 #dedication per periods
alpha=1.0 
beta=0.0
delta=0.0
args=""
j=0

while [[ $# > 0 ]]
do
    key="$1"
    case $key in
        -j|--job)
            shift
            j="$1"
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
            seed="$1"
            shift
        ;;
        *)
            args="$args$key "
            shift
        ;;
    esac
done

if hash condor_submit 2>/dev/null
then

HOME="/lhome/ext/iiia021/iiia0211"
ROOT_DIR="$HOME/YomaCR"
EXECUTABLE="$ROOT_DIR/solve.py"
LOG_DIR="$HOME/log/yoma/$m-dpp-$d"
DATA_DIR="$ROOT_DIR/data"
POOL_DIR="$DATA_DIR"

mkdir -p $LOG_DIR
STDOUT=$LOG_DIR/$j.stdout
STDERR=$LOG_DIR/$j.stderr
STDLOG=$LOG_DIR/$j.stdlog

tmpfile=$(mktemp)
condor_submit 1> $tmpfile <<EOF
universe = vanilla
stream_output = True
stream_error = True
executable = $EXECUTABLE
arguments = -i $POOL_DIR/$j.csv -s $seed -b $tb -g $gen $args
log = $STDLOG
output = $STDOUT
error = $STDERR
getenv = true
priority = $priority
queue
EOF

elif hash sbatch 2>/dev/null
then

USER=$(whoami)
BEEGFS="/mnt/beegfs/iiia/$USER"
ROOT_DIR="/home/$USER/YomaCR"
EXECUTABLE="$ROOT_DIR/solve.py"
LOG_DIR="$BEEGFS/yoma/$m-$d"
DATA_DIR="$ROOT_DIR/data"

mkdir -p $LOG_DIR
STDOUT=$LOG_DIR/$j.stdout
STDERR=$LOG_DIR/$j.stderr

tmpfile=$(mktemp)
sbatch 1> $tmpfile <<EOF
#!/bin/bash
#SBATCH --job-name=DPP-$m-$d-$j
#SBATCH --partition=quick
#SBATCH --time=10:30
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
srun modify_file $DATA_DIR/input_pref.csv $m $d $j
echo python3 $EXECUTABLE -m $m -d $d --alpha $alpha --beta $beta --delta $delta -j $j $args 1> $STDOUT
srun python3 $EXECUTABLE -m $m -d $d --alpha $alpha --beta $beta --delta $delta -j $j $args 1>> $STDOUT 2>> $STDERR
RET=\$?
exit \$RET
EOF

else
echo "Unknown cluster"
fi