# README #
Original README.md [link](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PaddlePaddle/Classification/RN50v1.5/README.md)

## Quick setup ##
1. Build docker image
```bash
docker build . -t nvidia_resnet50 --build-arg FROM_IMAGE_NAME=nvcr.io/nvidia/paddlepaddle:22.06-py3
```

2. Launch docker container
```bash
nvidia-docker run --rm -it -v <path to imagenet>:/imagenet --ipc=host nvidia_resnet50
```

3. Train
```bash
bash scripts/training/train_resnet50_AMP_QAT_10E_DGXA100.sh <path to AMP model>
```
If \<path to AMP model\> is /tmp, the data structure would be
```bash
/tmp/resnet_50_paddle.pdmodel
/tmp/resnet_50_paddle.pdiparams
/tmp/resnet_50_paddle.pdopt
```

4. Export
```bash
bash scripts/inference/export_resnet50_QAT.sh <path to preferred checkpoint>
```

5. Inference
```bash
bash scripts/inference/infer_resnet50_QAT.sh
```
