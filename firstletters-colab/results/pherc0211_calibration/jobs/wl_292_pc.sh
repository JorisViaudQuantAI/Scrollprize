# slot: cpu
# timeout: 3600
# post of the depth-sweep composites (sw0 single shift, swmax, swstitch, sworacleP/M) on dw035, dw033, control dw026
export WLSC=PHerc0211; source $INPUTS/wl_env.sh; export OMP_NUM_THREADS=3
mkdir -p $C/maps/comp && tar xzf $INPUTS/wl_comp.tgz -C $C/maps/comp
for n in dw035 dw033 dw026; do for d in sw0 swmax swstitch sworacleP sworacleM; do si_post $n $d $C/maps/comp/$n; done; done
