# Enhanced-Relationformer-for-Line-Detection

## Run baseline model (Relationformer)
1. **SCC Module Requirements:** cuda/12.2 python3/3.9.9 gcc/9.3.0

2. Build environment
    ```sh
    cd baseline_models/relatinoformer
    # create virtual environment
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    # compiling CUDA operators
    cd models/ops
    python setup.py build develop
    ```

3. Download data
    1. Using gdown
    ```sh
    gdown 1mgkUslw-tgT8EpS6ulRyA6gGy16IhXqc -O usgs.zip
    gdown 1Kd8TXYPv0LEAXC0NaaeSuKW_l5cBs605 -O uscities.zip
    ```
    2. Download from google drive
    **usgs data:** https://drive.google.com/file/d/1mgkUslw-tgT8EpS6ulRyA6gGy16IhXqc/view?usp=drive_link
    **uscities data:** https://drive.google.com/file/d/1Kd8TXYPv0LEAXC0NaaeSuKW_l5cBs605/view?usp=drive_link
    3. unzip and move to data folder
    ```sh
    unzip usgs.zip
    unzip uscities.zip

    mv train_data_g256_comb_less_neg_topo ./data
    mv 20cities ./data
    ```

4. Data preparation (for uscities data)
    ```sh
    python generate_data.py
    ```
    
5. Training
    1. Prepare config file
    The config file can be found at .configs/road_rgb_2D.yaml. Make custom changes if necessary.
    2. Training on multiple-GPU
    ```sh
    python train.py --config configs/{your_config_file} --cuda_visible_device 0 1
    ```