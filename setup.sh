source ~/.bashrc
conda create -n mmdet3d python=3.8
conda activate mmdet3d

conda install -y -c anaconda cmake
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install openmim
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
mim install mmdet
mim install mmsegmentation

pip uninstall -y mmdet3d
rm -rf ./build
pip install -e .

pip install -r requirements/optional.txt

pip install cumm-cu111
pip install spconv-cu111

pip install pyntcloud
pip install pyarrow
pip uninstall -y nuscenes-devkit
pip install setuptools==59.5.0
pip install nntime