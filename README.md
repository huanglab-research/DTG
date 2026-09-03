# Debiasing Text Guidance for Cross-Domain Few-Shot Object Detection

This is the implementation of paper: Debiasing Text Guidance for Cross-Domain Few-Shot Object Detection

> ⚠ **Note:** The source code is currently incomplete and will be fully released once the manuscript is accepted by the journal.

## Environment

This project is built on top of [mmdetection](https://github.com/open-mmlab/mmdetection). For detailed setup instructions, please refer to the official MMDetection [installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html).

```python
conda create --name repe python=3.8
conda activate repe
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

cd mmdetection
pip install -v -e .
pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

## Datasets:
You can download the datasets from the following links:  
[Baidu Netdisk, code:qmuk](https://pan.baidu.com/s/1gghXb9z3ebnSlSI3c5KJTQ?pwd=qmuk)  
**Note:**: If you use these datasets in your work, please make sure to cite them appropriately.

## Pretrained Model
[Grounding DINO-B](https://github.com/open-mmlab/mmdetection/tree/main/configs/grounding_dino)

## Training

To train the model:

1 x L40.

```python
bash ./mmdetection/tools/dist_train_muti.sh ./mmdetection/configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py
```

## Test

```python
bash ./mmdetection/tools/dist_test.sh ./mmdetection/configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py
```

## 📬 Contact

Feel free to contact me if there is any question. (Yanrui Zhao: [zhaoyanrui@stu.ouc.edu.cn](mailto:zhaoyanrui@stu.ouc.edu.cn), Lei Huang: [huangl@ouc.edu.cn](mailto:huangl@ouc.edu.cn))

---



