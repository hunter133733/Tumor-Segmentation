# Comparison of Segmentation and Foundation Models in MRI Imaging for Tumor Segmentation (CSFIM)
_A project for CMPT340 — Biomedical Computing at SFU Fall 2025._ 
_Sumant Dhir, Christopher Gutwin, Abdulahi Odeyeyiwa, Nick Smart, and Steven Shijia Zhang_

The goal of this project is to train a baseline segmentation model on MRI scans containing brain tumors, and compare it against a medical imaging foundation model by varying the input modality and training set completeness. By evaluating the models on performance, better diagnoses and patient care can be realized by choosing appropriate imaging modes and amounts for models deployed in healthcare settings.

## Table of Contents
1. [Demo Video](#demo)
2. [Installation](#install)
3. [Reproduction](#repro)
    - [Running the Trainer](#unet-trainer)
    - [Running the SAM Notebook](#sam)
4. [Important Links](#links)

<a name="demo"></a>
## Demo

[Demo Video](https://drive.google.com/file/d/1JcWgulfSrhyAGc0aGq8VX0-Mzu27QmLH/view)

<a name="install"></a>
## Installation
To install, create a new Python virtual environment using `python -m venv venv` and activate it with `source venv/bin/activate`. Running `pip install -r requirements.txt` will install necessary packages to run the scripts provided.

<a name="repro"></a>
## Reproduction

<a name="unet-trainer"></a>
### Running the nnUNet Trainer
Running the trainer requires the [BRaTS TCGA LGG Dataset](https://www.cancerimagingarchive.net/analysis-result/brats-tcga-lgg/) to be downloaded and extracted to the root of the repository. 
The script in `./src/FileAnalysis.py` will look for the `PKG - BraTS-TCGA-LGG` in the repository root, and extract files to their proper folders for nnUNet processing. Run this to setup the data for use with nnUNet.

#### Environment
The nnUNet training process requires environment variables to be set for processed, raw, and results data. Before continuing, set the following environment variables based on your shell:

##### Bash
```sh
export nnUNet_raw="./src/nnUNet_raw/nnUNet_raw"
export nnUNet_preprocessed="./src/nnUNet_raw/nnUNet_preprocessed"
export nnUNet_results="./src/nnUNet_raw/nnUNet_results"
```

##### Fish
```fish
set -gx nnUNet_raw "./src/nnUNet_raw/nnUNet_raw"
set -gx nnUNet_preprocessed "./src/nnUNet_raw/nnUNet_preprocessed"
set -gx nnUNet_results "./src/nnUNet_raw/nnUNet_results"
```

#### Preprocessing with nnUNet
Before training, run the preprocessor using the following command, ensuring you've set the environment variables as above.

```sh
nnUNetv2_plan_and_preprocess -d 501 --verify-dataset-integrity
```

This will place preprocessed files in the specified directory.

##### Verifying Preprocessing
To visually verify preprocessing, you can run the included project script to produce a figure of an MRI along with its segmentations.

```sh
python ./src/visualize_preprocessed_b2nd.py
```

#### Running Training

Running training is done on different subsets of data, specified by `splits_x.json` files moved to the preprocessed directory.
Subsets are as follows:
- 10% of data on all modes
- 25% of data on all modes
- 90% of data on all modes (the preconfigured split)

Each of the subset sizes represent `x` in `splits_x.json`. For example, `splits_10.json` runs 10% of the data for training.

Start training using the command
```sh
nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_10epochs
```

#### Predicting a Segmentation
Using the `imagesTs` folder, created by the `FileAnalysis.py` script, predictions can be made using the trained model with the `nnUNetv2_predict` command, specifying a location for results output.

In some cases, postprocessing can improve Dice scores by removing nonconnected segments. The `nnUNetv2_apply_postprocessing` command will try and improve scores.

#### Visualizing a Segmentation
Using the `visualize_segmentation_pred_subsets.py` script, you can visualize how well a segmentation stacks up to various training sizes. After following the training and setup cases in [Running the Trainer](#unet-trainer), the following command will output a figure showing the segmentation on a test case (ex. `DU-7014_0000` as an interesting case.)

![Visualize Segmentation Subsets Image Example](/assets/visualize_segmentation_pred_subsets.png)

```sh
python visualize_segmentation_pred_subsets.py \
--image /path/to/data/imagesTs/TCGA-DU-7014_0000.nii.gz \
--pred_10 /path/to/output/10/postprocessed/TCGA-DU-7014.nii.gz \
--pred_25 /path/to/output/25/postprocessed/TCGA-DU-7014.nii.gz \
--pred_90 /path/to/output/90/postprocessed/TCGA-DU-7014.nii.gz \
--axis 2 --slice-index 80 --output sc.png
```

<a name="sam"></a>
### Running SAM Notebook
The SAM experiments are run in **Google Colab** using 2D PNG slices and binary tumour masks.
#### Create slices + masks locally
Run on your own machine: 
```sh
python SamAnalysis.py
```
Make sure SamAnalysis.py is in the same folder as the **PKG - BraTS-TCGA-LGG** data folder so the paths in the py file work. Paths might need adjusting if the file setup is different than this. After running SamAnalysis.py, the results are stored in a folder called **sam_slices** in the same project folder.

#### Google Colab 
Open the SAM notebook **SAM(Roboflow_adaptation) (1).ipynb** in Google Colab.
In Google Colab, set **Runtime -> Change runtime type -> Hardware accelerator: GPU**. We used a T4 GPU.

#### Data Setup
Place the folder produced by SamAnalysis.py, in the first step, into your Google Drive. Then configure paths under the **Mount Drive** tab in the notebook to access the data in your Google Drive.

#### Results
Click **Run All** at the top left side of the screen and wait for all cells to complete running. It takes around 20 minutes for the notebook to complete running, where the majority of time is spent running SAM on each set of images.

<a name="links"></a>
## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/ETydZFuk0LpFoiaTlMgPoTABSCIO_YL-YbIvMBIAJ0MgRg) | [Slack channel](https://cmpt340fall2025.slack.com/archives/C09F8PS7UUU) | [Project report](https://google.com) |
|-----------|---------------|-------------------------|
