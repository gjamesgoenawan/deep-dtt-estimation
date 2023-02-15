# Deep Distance-to-Touchdown Prediction

<img src='images/normal_frame.png'></img>

Our Paper

An Implementation of research by [Dr. Duc-Thinh Pham](https://scholar.google.com.sg/citations?user=hrGEgUUAAAAJ&hl=en), [Gabriel James Goeawan](https://www.linkedin.com/in/gjamesgoenawan/),  and supervised by [Prof. Sameer Alam](https://scholar.google.com.sg/citations?user=5W6FyV0AAAAJ&hl=en) under Air Traffic Management Institute with and collaboration with Civil Aviation Authority of Singapore.

This study aims to estimate the distance-to-touchdown (DTD) in real time. Distance-to-touchdown is a critical parameter in final approach spacing and departure sequencing. Such capability in a Digital Tower environment can augment the runway controller’s sequencing and final approach spacing capabilities.
<br><br>

## Requirements
- Python 3.10
- PyTorch 1.13
- CUDA Accelerated GPU (TensorRT Optimization)
<br><br>

## Installation and Setup
This repository can be cloned using:
```
git clone https://github.com/gjamesgoenawan/deep-dtt-estimation.git
cd deep-dtt-estimation
```

All required dependencies can be installed using:
```
conda create --n "dtt_prediction" python=3.10 
conda activate dtt_prediction

pip install -r requirements.txt
```
If a specific PyTorch and CUDA version is needed, refer to [Pytorch Website](https://pytorch.org/) for installation details.
<br>

## Demo
A quick demo of pretrained model can be viewed in [`modeling_end2end.ipynb`](modeling_end2end.ipynb)
<br><br>

## Data Generation
As mentioned in our paper, we utilized [X-Plane 11 Flight Simulator](https://www.x-plane.com/) to generate dataset of various landing conditions and trajectories. For this purpose, we developed a python-based tool that is written for [XPPyython3](https://xppython3.readthedocs.io/en/latest/index.html). 

This tool can be viewed [here](xp11/PI_video.py).
<br><br>

## Documentations
Technical documentations regarding the implementation of this project can be accessed [here](#).
<br><br>

