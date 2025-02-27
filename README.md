# Endo-FASt3r
Endoscopic Foundation model Adaptation for Structure from motion

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
