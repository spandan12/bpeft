## Install environment

1. Install miniconda or anaconda in your device.

2. Run the following command to install all the environment. It requires `conda` and `pip`

    ``` sh env_setup.sh ```

3. Install the pre-trained models

    ``` 
    cd D_ALL 

    mkdir models

    wget -O ViT-B_16.npz https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
    
    wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth

    wget https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    ```

4. The few-shot data split for cifar10 and cifar100 are available in `D_ALL/fsDataset`. Similar split can be created for other datasets using script in `D_ALL/make_fs_initial_labeled_pool.py`

4. Run the following command to get started

    ``` sh  run.sh```

```
Change the variable name from run.sh to run different configuration.
 ```

6. Run the following command to get calibration


    ``` sh  run_calibration.sh```

