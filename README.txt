Run docker container as 
nvidia-docker run --restart=always --gpus all -d -it --name="SpaceHammer" -v /4tb:/4tb -v $(pwd):/SpaceHammer novorado-nvidia-cuda:1
