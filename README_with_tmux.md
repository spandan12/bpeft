## Install environment

1. Install miniconda or anaconda in your device.

2. Run the following command to install all the environment. It requires `conda` and `pip`

    - Create conda environment using python of 3.10.
        
         ``` conda create -n prompt python=3.10 ```

    - Activate the environment to install requirements inside the environment.
        
        ``` conda activate prompt ```

    - Install torch with cuda. 
        
        ``` python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 ```

    - Install all the necessary requirements.
        
        ``` python3 -m pip install tqdm pandas matplotlib seaborn scikit-learn scipy simplejson termcolor iopath fvcore timm==0.4.12 ml_collections```

3. Install the pre-trained models

    ``` 
    cd D_ALL 

    mkdir models

    cd models

    wget -O imagenet21k_ViT-B_16.npz https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

    cd ../../
    ```

4. The few-shot data split for cifar10 and cifar100 are available in `D_ALL/fsDataset`. Similar split can be created for other datasets using script in `D_ALL/make_fs_initial_labeled_pool.py`

4. Run the following command to get started

    ```
    Make sure to change ds_name to change dataset. Every new run requires to set new location. Make sure to change location. 
    ```    

    - Start new tmux window

        `tmux new`

    - Run the command to start training

        `CUDA_VISIBLE_DEVICES=<device_number> sh run.sh`
    
    - To exit the tmux window, 

        First press Keys `Control` + `B`

        Then press key `D` only
        
    - To re-enter tmux window, use tmux_window_number=0 if only one window present.

        tmux a -t <tmux_window_number>



6. Run the following command to get post calibration ECE and reliability plot

    Make sure to change the `base_path` with the path of the latest run.

    ``` python post_calibration```

