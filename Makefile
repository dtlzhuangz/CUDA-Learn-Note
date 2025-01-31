enter:
	sudo docker run --gpus all --shm-size 2g -v /mnt/local-storage/zhuangzhong/CUDA-Learn-Note:/data -it --rm nvcr.io/nvidia/pytorch:24.07-py3 bash