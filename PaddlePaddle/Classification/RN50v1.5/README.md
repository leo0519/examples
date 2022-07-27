# README #
Original README.md [link](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PaddlePaddle/Classification/RN50v1.5/README.md)

## Workaround and Issue ##
1. Duplicate code

The code in `utils/quant.py` is similar to PaddleSlim. They both call the same passes to modify the IR graph that is converted from the program.

```python
# utils/quant.py
def quant_aware(...):
    graph = IrGraph(core.Graph(program.desc), for_test=for_test)
    ...

def quant_convert(...):
    graph = IrGraph(core.Graph(program.desc), for_test=for_test)
    ...
```
Source: [utils/quant.py L1-148](https://github.com/leo0519/DeepLearningExamples/blob/paddle-qat/PaddlePaddle/Classification/RN50v1.5/utils/quant.py#L1-L148)

The QAT API can be integrated into Paddle and distributed optimizer, so the duplicate code can be avoided.

2. Build strategy

Some compilation strategies need to be manually open / closed.

```python
# program.py
if args.qat:
    build_strategy.fuse_bn_add_act_ops = False
    if is_train:
        build_strategy.num_trainers = get_num_trainers()
        build_strategy.trainer_id = get_trainer_id()

```
Source: [program.py L147-151](https://github.com/leo0519/DeepLearningExamples/blob/paddle-qat/PaddlePaddle/Classification/RN50v1.5/program.py#L147-L151)

Whether a compilation strategy is open or not can be determined by the distributed optimizer according to the flag of QAT.

3. Low level API

In `train.py`, we need to call a low level QAT API and compile the program again.
```python
# train.py
if args.qat:
    train_prog = quant_aware(train_prog, exe.place)
    train_prog._graph = program.compile_prog(args, train_prog, loss_name=train_fetchs['loss'][0].name, is_train=True)
    if eval_prog is not None:
        eval_prog = quant_aware(eval_prog, exe.place, for_test=True)

```
Source: [train.py L134-138](https://github.com/leo0519/DeepLearningExamples/blob/paddle-qat/PaddlePaddle/Classification/RN50v1.5/train.py#L134-L138)

These functions should be integrated into the distributed optimizer, so end-users can simply open the QAT flag to generate a QAT train program.

4. FP16 scale

We define a new function `convert_scale_fp16` to solve the FP16 scale problem, but it is hacking.
This conversion should be implemented in Paddle or even OP level.

Source: [export_model.py L134](https://github.com/leo0519/DeepLearningExamples/blob/paddle-qat/PaddlePaddle/Classification/RN50v1.5/train.py#L134) [utils/quant.py L150-166](https://github.com/leo0519/DeepLearningExamples/blob/paddle-qat/PaddlePaddle/Classification/RN50v1.5/utils/quant.py#L150-L166)

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

