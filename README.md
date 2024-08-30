## Prompt-Based Modality Alignment for Effective Multi-Modal Object Re-Identification
### Installation

```
conda create -n clipreid python=3.8
conda activate clipreid
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install yacs
pip install timm
pip install scikit-image
pip install tqdm
pip install ftfy
pip install regex
```

### Training

For example, if you want to run for the MSVR310, you need to modify the config file to

```
DATASETS:
  NAMES: ('MSVR310')
  ROOT_DIR: 'your_path'
OUTPUT_DIR: ('output/msvr310')
```

If you want to run PromptMA:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config_file configs/msvr310/vit.yml
```

### Evaluation

For example, if you want to test PromptMA for MSVR310

```
CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs/msvr310/vit.yml TEST.WEIGHT 'your_trained_checkpoints_path/ViT-B-16_120.pth'
```

### Weight
Mould-related weights files can be obtained from this [Google Drive](https://drive.google.com/drive/folders/1CqVSck0s_Cq0dwptvyRlPQ7_DRwRkYdo?usp=sharing)

### Acknowledgement

Codebase from [CLIP-ReID](https://github.com/Syliz517/CLIP-ReID), [TransReID](https://github.com/damo-cv/TransReID), [CLIP](https://github.com/openai/CLIP), and [CoOp](https://github.com/KaiyangZhou/CoOp).
