# SpeechTokenizer Scripts

## Encode MLS (Multilingual LibriSpeech) with SpeechTokenizer

tmux session `stok` should already be up. 

```sh
for i in {0..50}; do tmux new-window -t stok -n "block_$i" "srun --partition=a6000 --time=04:00:00 --gres=gpu:1 --qos=gpu-short python /mnt/scratch-artemis/anilkeshwani/SpeechTokenizer/scripts/mls.py $i 2>&1 | tee /mnt/scratch-artemis/anilkeshwani/stok-mls/logs/block_$i.log"; done
```
