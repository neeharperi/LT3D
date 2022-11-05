source ~/.bashrc

export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/Workspace/Core/nuscenes-lt3d/python-sdk"
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/Workspace/Core/av2-lt3d/src"

export PATH=/usr/lib/cuda/bin:$PATH
export CUDA_ROOT=/usr/lib/cuda
export CUDA_HOME=/usr/lib/cuda
export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH

wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sh Anaconda3-2021.11-Linux-x86_64.sh

conda create -n mmdet3d python=3.8
conda activate mmdet3d

conda install -y -c anaconda cmake
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge

pip3 install --upgrade pip

python3 -m pip install openmim
python3 -m pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
mim install mmdet
mim install mmsegmentation

python3 -m pip uninstall -y mmdet3d
rm -rf ./build
python3 -m pip install -e .

python3 -m pip install -r requirements/optional.txt

python3 -m pip install cumm-cu113
python3 -m pip install spconv-cu113

python3 -m pip install pyntcloud
python3 -m pip install pyarrow
python3 -m pip uninstall -y nuscenes-devkit
python3 -m pip install setuptools==59.5.0
python3 -m pip install nntime
