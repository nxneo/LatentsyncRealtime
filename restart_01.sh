kill $(lsof -t -i:28089)
cd /data0/projects/LatentSync

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/data0/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/data0/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/data0/anaconda3/etc/profile.d/conda.sh"
    else
	export PATH="/data0/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate latentsync
# export CUDA_VISIBLE_DEVICES=1
python infer_serv_0319.py
