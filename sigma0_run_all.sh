# default seeds: 1..10
seed_start=${1:-1}
seed_end=${2:-10}

logpath="logs/sigma0"
mkdir -p $logpath

nets=(bfc1 bafc1 afc1 afc2 afc3 fc4 bcv1 bcv2 bcv3 bcv6 bcv7)

for net in "${nets[@]}"; do
    for seed in $(seq $seed_start $seed_end); do
        logfile="$logpath/${net}.log"
        echo "$(date): Running: net=$net seed=$seed -> $logfile          " | tee -a "$logfile"
        python sigma_zero_attack.py --net "$net" -e 1 -tb 100 --step 100 --base_seed $seed >> "$logfile" 2>&1
    done
done

net=bcv4
for seed in $(seq $seed_start $seed_end); do
    logfile="$logpath/${net}_seed${seed}.log"
    echo "$(date): Running: net=$net seed=$seed -> $logfile          " | tee -a "$logfile"
    python sigma_zero_attack.py --net "$net" -e 1 -tb 10 --step 100 --base_seed $seed >> "$logfile" 2>&1
done