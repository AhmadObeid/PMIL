PMIL <img src="pmil-logo.png" width="280px" align="right" />
===========

***Overview:** PMIL is a versatile module for modeling object-level information in histopathology images through Topological Data Analysis. cPMIL is a cubic version that utilizes the multi-magnification factor in WSIs. The two modules can be plugged in most existing MIL pipelines with easy code manipulation, and offer an enhanced performance. In this repo, we demonstrate how to apply the modules to the [CLAM]() pipeline, for the classification task over the [PANDA]() challenge dataset*

<img src="model.gif" width="470px" align="center" />


[Installation](#installation) • [Patchifying and Feature Extraction](#patchifying-and-feature-extraction) • [Nuclei Extraction](#nuclei-extraction) • [Topological Data Extraction](#topological-data-extraction) • [TDA-Features](#TDA-features) • [Checkpoints](#Checkpoints) • [Codes Modification](#codes-modification) • [Examples](#examples) • [Cite](#reference)



## Updates:
* **01/03/2025**: Our cPMIL is submitted for revision.
* **02/08/2025**: Our PMIL model has been accepted at ISCAS 2025.


## Installation:
Clone this repo, and clone CLAM repo. The general recommendation is to create two environments, one for the topology analysis, and one for the CLAM pipeline.

To create the TDA environemnt with its essentials:
```
conda env create -f tda_env.yaml
```
To create the clam environemnt with its essentials:
```
conda env create -f clam_env.yaml
```

## Patchifying and Feature Extraction
Refer to the [CLAM](www.github.com/CLAM) for details about patchifying, and RGB feature extraction. We provide the csvs for reference.

## Nuclei Extraction
It is important to extract nuclei, as they represent the main key features in the histopathology image. The point cloud, represented by the extracted nuclei, will be the catalyst to start our TDA. Nuclei extraction can either be done using simple thresholding, or using a pretrained model like the [hovernet](www.github.com/hovernet) model. In this example, we used the former.
If a pretrained model is used, save the nuceli detection results in a json file (example included: nuc_example.json). This must be done for each patch in the WSI at a time.

We supply a profiling pipeline that facilitates an improved nuclei extraction based on thresholding and color convolution in displays/. By manual inspection, we extracted the minimum and maximum RGB values for each class in the PANDA dataset. For each given WSI, nuclei can be extracted based on these RGB values. This is done automatically in the code, so no need to worry about it. 
For the origianl method, you can follow [this](github.com/KitwareMedical/HistologyCancerDiagnosisDeepPersistenceHomology).


## Topological Data Extraction
For the topology analysis we follow the nice work of [Kitware](github.com/KitwareMedical/HistologyCancerDiagnosisDeepPersistenceHomology).
To extract the Persistance Images:
```
python -m generate_persistence...etc
```

If nuclei are extracted before strating TDA (i.e. through a pretrained model), they must be passed as argument.
otherwise...etc

## Codes Modification
### extract_features.py
We start by modifying the CLAM/extract_features.py file, which will be used to get the PI and PC features. The changes include:

1. Include two new model names in the --model_name argument. Namely, 'resnet50_trunc_3d', 'resnet50_trunc_PI', which extract PC and PI features.
2. Include a modality argument to separate PIs pipelines from PCs ones.
3. 

```
CUDA_VISIBLE_DEVICES=1 python extract_features.py \
--data_dir <dir_to_PIs> \
--csv_path <dir_to_csv> \
--feat_dir <output_dir> \
--batch_size 512 \
--slide_ext .tiff \
--model_name "resnet50_trunc_PI" \
--modality "PIs" \
```


## TDA-features
The TDA features are extracted in the same way the vanilla RGB features are extracted following [CLAM](www.github.com/CLAM). After going through the necessary code modification to facilitate handling PI images, both PI and PC features can be extracted and used in training.

## Checkpoints
For reproducability, all trained models used can be accessed [here](https://drive.google.com/drive/folders/1NZ82z0U_cexP6zkx1mRk-QeJyKWk4Q7z?usp=sharing).





### Examples

Please refer to our pre-print and [interactive demo](http://clam.mahmoodlab.org) for detailed results on three different problems and adaptability across data sources, imaging devices and tissue content. 

## Issues
- Please report all issues on the public forum.

## Funding


## Reference
If you ...:



```
@article{
}
```
