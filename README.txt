Run docker container as 
nvidia-docker run --restart=always --gpus all -d -it --name="SpaceHammer" -v /4tb:/4tb -v $(pwd):/SpaceHammer novorado-nvidia-cuda:1

p="/4tb/comma_ai.data/github/flownet2/models/FlowNet2-s/FlowNet2-S"
python3 caffe_weight_converter.py import/flownew_s.hdf5 ${p}_train.prototxt.template ${p}_weights.caffemodel
