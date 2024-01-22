# Dense Optical Tracking: Connecting the Dots


[Guillaume Le Moing](https://16lemoing.github.io/),
[Jean Ponce](https://www.di.ens.fr/~ponce/),
[Cordelia Schmid](https://thoth.inrialpes.fr/~schmid/)
<br>

#### [Project Page](https://16lemoing.github.io/dot) | [Paper](https://arxiv.org/abs/2312.00786) | [Video](https://www.youtube.com/watch?v=H0Rvq0OL87Y) | [BibTeX](#citation)

<p align="center"><img width="85%" src="assets/teaser.gif" /></p>

**DOT** unifies point tracking and optical flow techniques:
- It tracks all pixels in a frame simultaneously.
- It retains the robustness to occlusions and the accuracy of point tracking techniques.
- It enjoys the spatial consistency and runs at a comparable speed to optical flow techniques.

### News 📣
- [January 1st, 2024] We now support CoTracker2: DOT is up to 2x faster!


## Installation

### Set Up Environment
Clone the repository.
```
git clone https://github.com/16lemoing/dot
cd dot
```

<details>
<summary>Install dependencies.</summary>

[Optional] Create a conda environment.
```
conda create -n dot python=3.9
conda activate dot
```

Install the [PyTorch and TorchVision](https://pytorch.org/get-started/locally/) versions which are compatible with your CUDA configuration.
```
pip install torch==2.0.1 torchvision==0.15.2
```
Install inference dependencies.
```
pip install tqdm matplotlib einops scipy timm lmdb av mediapy
```

[Optional] Install training dependencies.
```
pip install tensorboard
```

[Optional] Set up custom modules from [PyTorch3D](https://github.com/facebookresearch/pytorch3d) to increase speed and reduce memory consumption of interpolation operations.
```
cd dot/utils/torch3d/ && python setup.py install && cd ../../..
```
</details>

### Download Checkpoints
```
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/cvo_raft_patch_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_raft_patch_4_alpha.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker_patch_4_wind_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker2_patch_4_wind_8.pth
```

## Demo

Download demo data.
```
wget -P datasets https://huggingface.co/16lemoing/dot/resolve/main/demo.zip
unzip datasets/demo.zip -d datasets/
```

### :spaghetti: - Spaghetti 

<details open>
<summary>Spaghetti with last frame (static).</summary>

Produce videos from the teaser figure.
```
python demo.py --visualization_modes spaghetti_last_static --video_path orange.mp4
python demo.py --visualization_modes spaghetti_last_static --video_path treadmill.mp4
```

https://github.com/16lemoing/dot/assets/32103788/82b888b8-3bc4-4ce8-9278-1adf39eb842a

```
python demo.py --visualization_modes spaghetti_last_static --video_path umbrella.mp4
```
</details>

<details>
<summary>Spaghetti from a mask on the first frame with first frame (static), and last frame.</summary>

https://github.com/16lemoing/dot/assets/32103788/088cc2d6-0cd3-449c-a42a-58f4371427d0

```
python demo.py --visualization_modes spaghetti_first_last_mask --video_path skateboard.mp4 --mask_path skateboard.png
```
</details>

### :rainbow: - Overlay 

<details open>
<summary>Overlay with tracks from all the pixels in the first frame.</summary>

https://github.com/16lemoing/dot/assets/32103788/7cc812c1-67fe-4710-9385-6675dd95cbf9

```
python demo.py --visualization_modes overlay --video_path cartwheel.mp4
```
</details>

<details>
<summary>Overlay with tracks reinitialized every 20 frames.</summary>

https://github.com/16lemoing/dot/assets/32103788/c0581f9a-0508-423b-b4db-9ec1aca2320a

```
python demo.py --visualization_modes overlay --video_path cartwheel.mp4 --inference_mode tracks_from_every_cell_in_every_frame --cell_time_steps 20
```
</details>

<details>
<summary>Overlay with tracks from a mask on the first frame with occluded regions marked as white stripes.</summary>

https://github.com/16lemoing/dot/assets/32103788/580ef7eb-4ae0-4174-9d62-9cfb62651c99

```
python demo.py --visualization_modes overlay_stripes_mask --video_path varanus.mp4 --mask_path varanus.png
```
</details>

## Evaluation

### Data Preprocessing

<details>
<summary>Download Kubric-CVO test data.</summary>

```
wget -P datasets/kubric/cvo https://huggingface.co/datasets/16lemoing/cvo/resolve/main/cvo_test.lmdb
wget -P datasets/kubric/cvo https://huggingface.co/datasets/16lemoing/cvo/resolve/main/cvo_test_extended.lmdb
```
</details>

<details>
<summary>Download TAP test data.</summary>

```
wget -P datasets/tap https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
wget -P datasets/tap https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip
wget -P datasets/tap https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip
unzip "datasets/tap/*.zip" -d datasets/tap/
```
</details>


### Compute Evaluation Metrics
```
python test_cvo.py --split {clean|final|extended}
python test_tap.py --split {davis|kinetics|rgb_stacking}
```

### Benchmarking

Results reproduced with this codebase on **Kubric-CVO**.

<details>
<summary>Detailed metrics.</summary>

We compute the dense motion between the first and last frames of videos and report:
* the end point error (EPE) of flows
* the intersection over union (IoU) of occluded regions
* the average inference time per video (in seconds) on a NVIDIA V100 GPU
</details>

<details>
<summary>Command line for each method.</summary>

```
python test_cvo.py --split {final|extended} --model pt --tracker_config configs/cotracker_patch_4_wind_8.json --tracker_path checkpoints/movi_f_cotracker_patch_4_wind_8.pth
python test_cvo.py --split {final|extended} --model pt --tracker_config configs/cotracker2_patch_4_wind_8.json --tracker_path checkpoints/movi_f_cotracker2_patch_4_wind_8.pth
python test_cvo.py --split {final|extended} --model dot --tracker_config configs/cotracker_patch_4_wind_8.json --tracker_path checkpoints/movi_f_cotracker_patch_4_wind_8.pth
python test_cvo.py --split {final|extended} --model dot --tracker_config configs/cotracker2_patch_4_wind_8.json --tracker_path checkpoints/movi_f_cotracker2_patch_4_wind_8.pth
```
</details>

| Method                   | EPE &darr;         | IoU &uarr;         | Time &darr;         |
|:-------------------------|:------------------:|:------------------:|:-------------------:|
| CoTracker                | 1.45 / 5.10        | 75.0 / 70.3        | 177 / 1289          |
| CoTracker2               | 1.47 / 5.45        | 77.9 / 69.2        | 80.2 / 865          |
| DOT* (Cotracker + RAFT)  | 1.38 / **4.97**    | 80.2 / **71.2**    | 1.57 / 10.3         |
| DOT* (Cotracker2 + RAFT) | **1.37** / 5.11    | **80.3** / 71.1    | **0.82** / **7.02** |
|                          | _final / extended_ | _final / extended_ | _final / extended_  |

_* results obtained using N=2048 initial tracks, other speed / performance trade-offs are possible by using different values for N._

## Training

### Data Preprocessing

<details>
<summary>Download preprocessed data.</summary>

Download Kubric-MoviF train data.
```
wget -P datasets/kubric/movi_f https://huggingface.co/datasets/16lemoing/movi_f/resolve/main/video_part.zip
wget -P datasets/kubric/movi_f https://huggingface.co/datasets/16lemoing/movi_f/resolve/main/video_part.z01
wget -P datasets/kubric/movi_f https://huggingface.co/datasets/16lemoing/movi_f/resolve/main/ground_truth.zip
wget -P datasets/kubric/movi_f https://huggingface.co/datasets/16lemoing/movi_f/resolve/main/cotracker.zip
```

Unzip data.
```
zip -F video_part.zip --out datasets/kubric/movi_f/video.zip
unzip datasets/kubric/movi_f/video.zip -d datasets/kubric/movi_f/
unzip datasets/kubric/movi_f/ground_truth.zip -d datasets/kubric/movi_f/
unzip datasets/kubric/movi_f/cotracker.zip -d datasets/kubric/movi_f/
```
</details>

<details>
<summary>Or run preprocessing steps yourself.</summary>

Install additional dependencies.
```
pip install joblib tensorflow tensorflow_datasets tensorflow-graphics
```

Download Kubric-MoviF train data.
```
python preprocess.py --extract_movi_f
```
[Requires a GPU] Save tracks from CoTracker for Kubric-MoviF train data.
```
python preprocess.py --save_tracks
```
</details>



### Optimization

```
python train.py
```

## License

Most of our code is licensed under the MIT License.
However, some parts of the code are adapted from external sources and conserve their original license:
CoTracker is licensed under CC-BY-NC,
RAFT uses the BSD 3-Clause License,
Kubric and TAP use the Apache 2.0 License,
and PyTorch3D is licensed under a BSD License.

## Contributing

We actively encourage contributions. Want to feature a cool application which builds upon DOT, or add support to another point tracker / optical flow model? Don't hesitate to open an issue to discuss about it.

## Acknowledgments

We want to thank [CoTracker](https://github.com/facebookresearch/co-tracker), [RAFT](https://github.com/princeton-vl/RAFT), [AccFlow](https://github.com/mulns/AccFlow), [TAP](https://github.com/google-deepmind/tapnet), and [Kubric](https://github.com/google-research/kubric) for publicly releasing their code, models and data.

## Citation
Please note that any use of the code in a publication must explicitly refer to:
```
@article{lemoing2023dense,
  title={Dense Optical Tracking: Connecting the Dots},
  author={Guillaume Le Moing and Jean Ponce and Cordelia Schmid},
  journal={arXiv preprint},
  year={2023}
}
```
