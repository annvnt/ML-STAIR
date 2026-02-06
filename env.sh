#conda create -n stair python=3.9;conda activate stair;bash env.sh

pip install freerec==0.8.5
# Pin numpy to latest 1.x
conda install "numpy<2.0" -y
conda install pytorch=2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
conda install pandas tqdm matplotlib torchdata=0.7.1 pyg -c pyg