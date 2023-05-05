# TransESC

> The official implementation for the Findings of ACL 2023 paper *TransESC: Smoothing Emotional Support Conversation via Turn-Level State Transition*.

<img src="https://img.shields.io/badge/Venue-ACL--23-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Issues-Welcome-red">

## Requirements
* Python 3.7
* PyTorch 1.8.2
* Transformers 4.12.3
* CUDA 11.1

## Preparation

Download the processed dataset, annotated emotion and keywords, and generated commonsense knowledge from https://drive.google.com/drive/folders/1rNpa5vDuB7KssQuVBAhkKtmAuqPa0Qgf?usp=sharing. Then place them in `/dataset`.

## Training

```sh
python main.py
```

## Evaluation and Generation

```sh
python main.py --test 
```