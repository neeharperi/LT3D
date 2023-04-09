source ~/.bashrc

conda install -y -c anaconda cmake
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install mmdet==2.25.2
pip install mmsegmentation==0.28.0

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
