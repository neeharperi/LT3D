source ~/.bashrc

conda install -y -c anaconda cmake
conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install openmim
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
mim install mmdet
mim install mmsegmentation

pip uninstall -y mmdet3d
rm -rf ./build
pip install -e .

pip install -r requirements/optional.txt

pip install cumm-cu113
pip install spconv-cu113

pip install pyntcloud
pip install pyarrow
pip uninstall -y nuscenes-devkit
pip install setuptools==59.5.0
pip install nntime
