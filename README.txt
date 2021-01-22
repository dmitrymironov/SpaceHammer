Run docker container as 
nvidia-docker run --restart=always --gpus all -d -it --name="SpaceHammer" -v /4tb:/4tb -v $(pwd):/SpaceHammer novorado-nvidia-cuda:1

p="/4tb/comma_ai.data/github/flownet2/models/FlowNet2-s/FlowNet2-S"
txt=${p}_train.prototxt.template 
wgt=${p}_weights.caffemodel
python3 caffe_weight_converter.py import/flownew_s.hdf5 /SpaceHammer/FlowNet2-S_train.prototxt ${p}_weights.caffemodel

# Copy protxt before editing
cp /4tb/comma_ai.data/github/flownet2/models/FlowNet2-s/FlowNet2-S_train.prototxt.template FlowNet2-S_train.prototxt 
