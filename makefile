create-env:
	conda create -n sglang python=3.12 -y

install-torch:
	# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
	conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch

install-sglang:
	pip install --upgrade pip
	pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

run:
	python3 run.py

clean:
	conda remove pytorch torchvision torchaudio pytorch-cuda -y

save:
	git-save

run2:
	set -x CUDA_HOME ~/anaconda3/envs/sglang
	set -x CUDA_PATH ~/anaconda3/envs/sglang
	set -x CUDACXX ~/anaconda3/envs/sglang/bin/nvcc
	set -x LD_LIBRARY_PATH ~/anaconda3/envs/sglang/lib:~/anaconda3/envs/sglang/lib64:$LD_LIBRARY_PATH
	make run