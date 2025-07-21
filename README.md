# DMOSpeech 2: Reinforcement Learning for Duration Prediction in Metric-Optimized Speech Synthesis

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/yl4579/DMOSpeech2)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://dmospeech2.github.io/)

### Yinghao Aaron Li*, Xilin Jiang*, Fei Tao, Cheng Niu, Kaifeng Xu, Juntong Song, Nima Mesgarani

> Diffusion-based text-to-speech (TTS) systems have made remarkable progress in zero-shot speech synthesis, yet optimizing all components for perceptual metrics remains challenging. Prior work with DMOSpeech demonstrated direct metric optimization for speech generation components, but duration prediction remained unoptimized. This paper presents DMOSpeech 2, which extends metric optimization to the duration predictor through a reinforcement learning approach. The proposed system implements a novel duration policy framework using group relative preference optimization (GRPO) with speaker similarity and word error rate as reward signals. By optimizing this previously unoptimized component, DMOSpeech 2 creates a more complete metric-optimized synthesis pipeline. Additionally, this paper introduces teacher-guided sampling, a hybrid approach leveraging a teacher model for initial denoising steps before transitioning to the student model, significantly improving output diversity while maintaining efficiency. Comprehensive evaluations demonstrate superior performance across all metrics compared to previous systems, while reducing sampling steps by half without quality degradation. These advances represent a significant step toward speech synthesis systems with metric optimization across multiple components.

*: Equal contribution

## Pre-requisites

### Create a separate environment if needed

```bash
conda create -n dmo2 python=3.10
conda activate dmo2
```

### Install required packages

1. Clone this repository:
```bash
git clone https://github.com/yl4579/DMOSpeech2.git
cd DMOSpeech2
```
2. Install python requirements: 
```bash
pip install -r requirements.txt
```

Alternatively, you can also create a [F5-TTS enviornment](https://github.com/SWivid/F5-TTS) and directy run the inference with it. 

## Inference

Under construction

## Training

Under construction

## References

- [F5-TTS](https://github.com/SWivid/F5-TTS): Main codebase modified from F5-TTS repo, which also serves as the teacher
- [DMD2](https://github.com/tianweiy/DMD2): Training recipe
- [simple_GRPO](https://github.com/lsdefine/simple_GRPO): GRPO training recipe
