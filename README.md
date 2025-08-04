# TTA

# setup
```
pip3 install transformers datasets accelerate evaluate
pip3 install torch torchvision torchaudio
pip install pycocoevalcap
pip install torchcodec
conda install -n tta_env ipykernel --update-deps --force-reinstall
cd ~/.conda/envs/tta_env
mv ./lib/libstdc++.so.6 ./lib/libstdc++.so.6.old
conda install libstdcxx-ng>=12.2.0
pip install soundfile
pip install jiwer
pip install tensorboard
git clone https://github.com/maxschelski/pytorch-cluster-metrics.git
cd pytorch-cluster-metrics
pip install -e .
pip install scikit-learn
```
