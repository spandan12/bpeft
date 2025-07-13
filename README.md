## Install environment

1. Install miniconda or anaconda in your device.

2. Run the following command to install all the environment. It requires `conda` and `pip`

    ``` sh env_setup.sh ```

    Activate the environment

    ``` conda activate prompt ```

3. Install the pre-trained models

    ``` 
    cd D_ALL 

    mkdir models

    cd models

    wget -O imagenet21k_ViT-B_16.npz https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

    cd ../..
    ```

4. The few-shot data split for cifar10 and cifar100 are available in `D_ALL/fsDataset`. Similar split can be created for other datasets using script in `D_ALL/make_fs_initial_labeled_pool.py`

4. Run the following command to get started

```
Make sure to change ds_name to change dataset. Every new run requires to set new location. Make sure to change location. 
 ```    


`sh submit.sh`


6. Run the following command to get post calibration ECE and reliability plot

    Make sure to change the `base_path` with the path of the latest run.

    ``` python post_calibration```

