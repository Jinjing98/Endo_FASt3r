

[comment]: <> (# Endo-FASt3r: Endoscopic Foundation model Adaptation for Structure from motion)



  <p align="center">
  <h1 align="center">Endo-FASt3r: Endoscopic Foundation model Adaptation for Structure from motion</h1>
  <p align="center">
    <strong>Mona Sheikh Zeinoddin</strong>
    ·
    <strong>Mobarak I. Hoque</strong>
    ·
    <strong>Zafer Tandogdu</strong>
    ·
    <strong>Greg L. Shaw</strong>
    ·
    <strong>Matthew J. Clarkson</strong>
    ·
    <strong>Evangelos B. Mazomenos</strong>
    ·
    <strong>Danail Stoyanov</strong>
  </p>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/pdf/2503.07204v1">Paper</a></h3>
  <div align="center"></div>

## SCARED training

```shell
python train_end_to_end.py --data_path <your_data_path> --log_dir <path_to_save_model (depth, pose, appearance flow, optical flow)>
```

## Evaluation on SCARED
Depth Evaluation:
```shell
python evaluate_depth.py --data_path <your_data_path> --load_weights_folder <path_to_weights_i_folder> --eval_mono
```
Pose evaluation:
```shell
python evaluate_pose.py --data_path <your_data_path>  --load_weights_folder <path_to_weights_i_folder> --scared_pose_seq <trajectory_1_or_2>
```    
