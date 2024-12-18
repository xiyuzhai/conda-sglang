create-env:
	conda create -n sglang python=3.12 -y

install-torch:
	conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

install-sglang:
	pip install --upgrade pip
	pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/

run:
	python run.py
