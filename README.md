# Endo-FASt3r
Endoscopic Foundation model Adaptation for Structure from motion

## ⏳ Endovis training


```shell
CUDA_VISIBLE_DEVICES=0 python train_end_to_end.py --data_path <your_data_path> --log_dir <path_to_save_model (depth, pose, appearance flow, optical flow)>
```

## 📊 Endovis evaluation

To prepare the ground truth depth maps run:
```shell
CUDA_VISIBLE_DEVICES=0 python export_gt_depth.py --data_path endovis_data --split endovis
```
...assuming that you have placed the endovis dataset in the default location of `./endovis_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --data_path <your_data_path> --load_weights_folder ~/mono_model/mdp/models/weights_19 --eval_mono
```
