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
- [November 19th, 2024] We now support CoTracker3! Thanks [@yindaheng98](https://www.github.com/yindaheng98) for the contribution.
- [March 1st, 2024] We now support TAPIR and BootsTAPIR: new SOTA on DAVIS!
- [February 26th, 2024] DOT has been accepted to [CVPR 2024](https://cvpr.thecvf.com)!
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
pip install tqdm matplotlib einops einshape scipy timm lmdb av mediapy
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

## Model Zoo

### Optical flow estimation
&rarr; *find motion boundaries*

<table>
  <tr>
    <th colspan="1">Model</th>
    <th colspan="1">Data</th>
    <th colspan="2">Download</th>
  </tr>
  <tr>
    <td>RAFT</td>
    <td>Kubric-CVO</td>
    <td><a href="https://huggingface.co/16lemoing/dot/resolve/main/cvo_raft_patch_8.pth">Checkpoint</a></td>
    <td><a href="https://github.com/16lemoing/dot/blob/main/configs/raft_patch_8.json">Config</a></td>
  </tr>
</table>

### Point tracking initialization
&rarr; *track sparse queries, half at motion boundaries, half randomly*

<table>
  <tr>
    <th colspan="1">Model</th>
    <th colspan="1">Data</th>
    <th colspan="2">Download</th>
  </tr>
  <tr>
    <td>CoTracker</td>
    <td>Kubric-MOViF</td>
    <td><a href="https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker_patch_4_wind_8.pth">Checkpoint</a></td>
    <td><a href="https://github.com/16lemoing/dot/blob/main/configs/cotracker_patch_4_wind_8.json">Config</a></td>
  </tr>
  <tr>
    <td>CoTracker2</td>
    <td>Kubric-MOViF</td>
    <td><a href="https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker2_patch_4_wind_8.pth">Checkpoint</a></td>
    <td><a href="https://github.com/16lemoing/dot/blob/main/configs/cotracker2_patch_4_wind_8.json">Config</a></td>
  </tr>
  <tr>
    <td>CoTracker3</td>
    <td>Kubric-MOViF + Real data</td>
    <td><a href="https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth">Checkpoint</a></td>
    <td><a href="https://github.com/16lemoing/dot/blob/main/configs/cotracker3_wind_60.json">Config</a></td>
  </tr>
  <tr>
    <td>TAPIR</td>
    <td>Kubric-Panning-MOViE</td>
    <td><a href="https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_tapir.pth">Checkpoint</a></td>
    <td><a href="https://github.com/16lemoing/dot/blob/main/configs/tapir.json">Config</a></td>
  </tr>
  <tr>
    <td>BootsTAPIR</td>
    <td>Kubric-Panning-MOViE + Real data</td>
    <td><a href="https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_plus_bootstapir.pth">Checkpoint</a></td>
    <td><a href="https://github.com/16lemoing/dot/blob/main/configs/bootstapir.json">Config</a></td>
  </tr>
</table>

### Optical flow refinement
&rarr; *get dense motion from sparse point tracks*

<table>
  <tr>
    <th colspan="1">Model</th>
    <th colspan="1">Input</th>
    <th colspan="1">Data</th>
    <th colspan="2">Download</th>
  </tr>
  <tr>
    <td>RAFT</td>
    <td>CoTracker</td>
    <td>Kubric-MOViF</td>
    <td><a href="https://huggingface.co/16lemoing/dot/resolve/main/movi_f_raft_patch_4_alpha.pth">Checkpoint</a></td>
    <td><a href="https://github.com/16lemoing/dot/blob/main/configs/raft_patch_4_alpha.json">Config</a></td>
  </tr>
</table>

<details>
<summary>Command line to download all checkpoints.</summary>

```
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/cvo_raft_patch_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_raft_patch_4_alpha.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker_patch_4_wind_8.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/movi_f_cotracker2_patch_4_wind_8.pth
wget -O checkpoints/movi_f_cotracker3_wind_60.pth https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_tapir.pth
wget -P checkpoints https://huggingface.co/16lemoing/dot/resolve/main/panning_movi_e_plus_bootstapir.pth
```
</details>

## Demo

Download demo data.
```
wget -P datasets https://huggingface.co/16lemoing/dot/resolve/main/demo.zip
unzip datasets/demo.zip -d datasets/
```

### Spaghetti :spaghetti:

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

###  Overlay :rainbow:

<details open>
<summary>Overlay with tracks from all the pixels in the first frame.</summary>

https://github.com/16lemoing/dot/assets/32103788/7cc812c1-67fe-4710-9385-6675dd95cbf9

```
python demo.py --visualization_modeovis overlay --video_path cartwheel.mp4
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

## Benchmarking

### Kubric-CVO

<details>
<summary>Detailed metrics.</summary>

We compute the dense motion between the first and last frames of videos, and report:
* the end point error (EPE) of flows
* the intersection over union (IoU) of occluded regions
* the average inference time per video (in seconds) on a NVIDIA V100 GPU
</details>

<details>
<summary>Command line for each method.</summary>

```
python test_cvo.py --split {final|extended} --model pt --tracker_config configs/tapir.json --tracker_path checkpoints/panning_movi_e_tapir.pth
python test_cvo.py --split {final|extended} --model pt --tracker_config configs/bootstapir.json --tracker_path checkpoints/panning_movi_e_plus_bootstapir.pth
python test_cvo.py --split {final|extended} --model pt --tracker_config configs/cotracker_patch_4_wind_8.json --tracker_path checkpoints/movi_f_cotracker_patch_4_wind_8.pth
python test_cvo.py --split {final|extended} --model pt --tracker_config configs/cotracker2_patch_4_wind_8.json --tracker_path checkpoints/movi_f_cotracker2_patch_4_wind_8.pth
python test_cvo.py --split {final|extended} --model pt --tracker_config configs/cotracker3_wind_60.json --tracker_path checkpoints/movi_f_cotracker3_wind_60.pth
python test_cvo.py --split {final|extended} --model dot --tracker_config configs/cotracker_patch_4_wind_8.json --tracker_path checkpoints/movi_f_cotracker_patch_4_wind_8.pth
python test_cvo.py --split {final|extended} --model dot --tracker_config configs/cotracker2_patch_4_wind_8.json --tracker_path checkpoints/movi_f_cotracker2_patch_4_wind_8.pth
python test_cvo.py --split {final|extended} --model dot --tracker_config configs/cotracker3_wind_60.json --tracker_path checkpoints/movi_f_cotracker3_wind_60.pth
```
</details>

<details>
<summary>Comments.</summary>
We run DOT using N=2048 initial tracks, other speed / performance trade-offs are possible by using different values for N.
</details>

<hr>

<details open>
<summary>CVO.</summary>

<table>
  <tr>
    <th colspan="1" rowspan="2">Method</th>
    <th colspan="3">Final</th>
    <th colspan="3">Extended</th>
  </tr>
  <tr>
    <td>EPE &darr;</td>
    <td>IoU &uarr;</td>
    <td>Time &darr;</td>
    <td>EPE &darr;</td>
    <td>IoU &uarr;</td>
    <td>Time &darr;</td>
  </tr>
  <tr>
    <td>TAPIR</td>
    <td>4.59</td>
    <td>73.8</td>
    <td>129</td>
    <td>22.6</td>
    <td>68.6</td>
    <td>811</td>
  </tr>
  <tr>
    <td>BootsTAPIR</td>
    <td>4.17</td>
    <td>74.9</td>
    <td>142</td>
    <td>25.3</td>
    <td>68.1</td>
    <td>892</td>
  </tr>
  <tr>
    <td>CoTracker</td>
    <td>1.45</td>
    <td>75.0</td>
    <td>177</td>
    <td>5.10</td>
    <td>70.3</td>
    <td>1289</td>
  </tr>
  <tr>
    <td>CoTracker2</td>
    <td>1.47</td>
    <td>77.9</td>
    <td>80.2</td>
    <td>5.45</td>
    <td>69.2</td>
    <td>865</td>
  </tr>
  <tr>
    <td>DOT (CoTracker + RAFT)</td>
    <td><ins>1.38</ins></td>
    <td><ins>80.2</ins></td>
    <td><ins>1.57</ins></td>
    <td><b>4.97</b></td>
    <td><b>71.2</b></td>
    <td><ins>10.3</ins></td>
  </tr>
  <tr>
    <td>DOT (CoTracker2 + RAFT)</td>
    <td><b>1.37</b></td>
    <td><b>80.3</b></td>
    <td><b>0.82</b></td>
    <td><ins>5.11</ins></td>
    <td><ins>71.1</ins></td>
    <td><b>7.02</b></td>
  </tr>
</table>
</details>

### TAP

<details>
<summary>Detailed metrics.</summary>

We compute the dense motion between the query frames (query first mode) and every other frame of videos, and report for ground truth trajectories:
* the average jaccard (AJ)
* the average proportion of points within a threshold (&lt;&delta;)
* the occlusion accuracy (OA)
* the average inference time per video (in seconds) on a NVIDIA V100 GPU
</details>

<details>
<summary>Command line for each method.</summary>

```
python test_tap.py --split {davis|rgb_stacking} --query_mode {first|strided} --model dot --tracker_config configs/cotracker_patch_4_wind_8.json --tracker_path checkpoints/movi_f_cotracker_patch_4_wind_8.pth
python test_tap.py --split {davis|rgb_stacking} --query_mode {first|strided} --model dot --tracker_config configs/cotracker2_patch_4_wind_8.json --tracker_path checkpoints/movi_f_cotracker2_patch_4_wind_8.pth
python test_tap.py --split {davis|rgb_stacking} --query_mode {first|strided} --model dot --tracker_config configs/cotracker3_wind_60.json --tracker_path checkpoints/movi_f_cotracker3_wind_60.pth
python test_tap.py --split {davis|rgb_stacking} --query_mode {first|strided} --model dot --tracker_config configs/tapir.json --tracker_path checkpoints/panning_movi_e_tapir.pth
python test_tap.py --split {davis|rgb_stacking} --query_mode {first|strided} --model dot --tracker_config configs/bootstapir.json --tracker_path checkpoints/panning_movi_e_plus_bootstapir.pth
```
</details>

<details>
<summary>Comments.</summary>
We run DOT using N=8192 initial tracks, other speed / performance trade-offs are possible by using different values for N.
Here, TAPIR and BootsTAPIR are faster than CoTracker and CoTracker2 since they operate directly at 256x256 resolution while the latter resize videos to a higher resolution.
</details>

<hr>

<details open>
<summary>DAVIS.</summary>
<table>
  <tr>
    <th colspan="1" rowspan="2">Method</th>
    <th colspan="4">First</th>
    <th colspan="4">Strided</th>
  </tr>
  <tr>
    <td>AJ &uarr;</td>
    <td>OA &uarr;</td>
    <td>&lt;&delta; &uarr;</td>
    <td>Time &darr;</td>
    <td>AJ &uarr;</td>
    <td>OA &uarr;</td>
    <td>&lt;&delta; &uarr;</td>
    <td>Time &darr;</td>
  </tr>
  <tr>
    <th colspan="9">Published results</th>
  </tr>
  <tr>
    <td>OmniMotion</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>51.7</td>
    <td>85.3</td>
    <td>67.5</td>
    <td>~32400</td>
  </tr>
  <tr>
    <td>DinoTracker</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>62.3</td>
    <td>87.5</td>
    <td>78.2</td>
    <td>~5760</td>
  </tr>
  <tr>
    <th colspan="9">This repo's results</th>
  </tr>
  <tr>
    <td>DOT (CoTracker + RAFT)</td>
    <td>61.2</td>
    <td>88.8</td>
    <td>74.9</td>
    <td>85.7</td>
    <td>66.1</td>
    <td>90.0</td>
    <td>79.4</td>
    <td>131</td>
  </tr>
  <tr>
    <td>DOT (CoTracker2 + RAFT)</td>
    <td>61.2</td>
    <td><ins>89.7</ins></td>
    <td>75.3</td>
    <td>99.1</td>
    <td>67.7</td>
    <td>91.2</td>
    <td>80.6</td>
    <td>141</td>
  </tr>
  <tr>
    <td>DOT (TAPIR + RAFT)</td>
    <td><ins>61.6</ins></td>
    <td>89.5</td>
    <td><ins>75.4</ins></td>
    <td><b>39.5</b></td>
    <td><ins>67.3</ins></td>
    <td><ins>91.0</ins></td>
    <td><ins>79.9</ins></td>
    <td><ins>88.9</ins></td>
  </tr>
  <tr>
    <td>DOT (BootsTAPIR + RAFT)</td>
    <td><b>62.8</b></td>
    <td><b>90.2</b></td>
    <td><b>76.8</b></td>
    <td><ins>42.3</ins></td>
    <td><b>68.5</b></td>
    <td><b>91.7</b></td>
    <td><b>81.3</b></td>
    <td><b>90.6</b></td>
  </tr>
</table>
</details>

<details>
<summary>RGB-Stacking.</summary>
<table>
  <tr>
    <th colspan="1" rowspan="2">Method</th>
    <th colspan="4">First</th>
    <th colspan="4">Strided</th>
  </tr>
  <tr>
    <td>AJ &uarr;</td>
    <td>OA &uarr;</td>
    <td>&lt;&delta; &uarr;</td>
    <td>Time &darr;</td>
    <td>AJ &uarr;</td>
    <td>OA &uarr;</td>
    <td>&lt;&delta; &uarr;</td>
    <td>Time &darr;</td>
  </tr>
  <tr>
    <th colspan="9">Published results</th>
  </tr>
  <tr>
    <td>OmniMotion</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>77.5</td>
    <td>93.5</td>
    <td>87.0</td>
    <td>~32400</td>
  </tr>
  <tr>
    <td>DinoTracker</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <th colspan="9">This repo's results</th>
  </tr>
  <tr>
    <td>DOT (CoTracker + RAFT)</td>
    <td><b>77.2</b></td>
    <td><b>93.3</b></td>
    <td><b>87.7</b></td>
    <td>270</td>
    <td><b>83.5</b></td>
    <td><b>95.7</b></td>
    <td><b>91.4</b></td>
    <td>1014</td>
  </tr>
  <tr>
    <td>DOT (CoTracker2 + RAFT)</td>
    <td><ins>77.2</ins></td>
    <td><ins>92.6</ins></td>
    <td><ins>87.1</ins></td>
    <td>330</td>
    <td><ins>83.2</ins></td>
    <td><ins>95.3</ins></td>
    <td><ins>91.0</ins></td>
    <td>1074</td>
  </tr>
  <tr>
    <td>DOT (TAPIR + RAFT)</td>
    <td>65.7</td>
    <td>89.1</td>
    <td>81.9</td>
    <td><b>105</b></td>
    <td>74.6</td>
    <td>93.4</td>
    <td>86.4</td>
    <td><b>843</b></td>
  </tr>
  <tr>
    <td>DOT (BootsTAPIR + RAFT)</td>
    <td>71.0</td>
    <td>90.7</td>
    <td>85.2</td>
    <td><ins>112</ins></td>
    <td>79.7</td>
    <td>94.7</td>
    <td>89.6</td>
    <td><ins>852</ins></td>
  </tr>
</table>
</details>

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
zip -F datasets/kubric/movi_f/video_part.zip --out datasets/kubric/movi_f/video.zip
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

We want to thank [RAFT](https://github.com/princeton-vl/RAFT), [AccFlow](https://github.com/mulns/AccFlow), [TAP](https://github.com/google-deepmind/tapnet), [CoTracker](https://github.com/facebookresearch/co-tracker), [DinoTracker](https://github.com/AssafSinger94/dino-tracker), [OmniMotion](https://github.com/qianqianwang68/omnimotion) and [Kubric](https://github.com/google-research/kubric) for publicly releasing their code, models and data.

## Citation
Please note that any use of the code in a publication must explicitly refer to:
```
@inproceedings{lemoing2024dense,
  title = {Dense Optical Tracking: Connecting the Dots},
  author = {Le Moing, Guillaume and Ponce, Jean and Schmid, Cordelia},
  year = {2024},
  booktitle = {CVPR}
}
```
