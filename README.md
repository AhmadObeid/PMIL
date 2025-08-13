PMIL 
===========

***Overview:** PMIL is a versatile module for modeling object-level information in histopathology images through Topological Data Analysis. cPMIL is a cubic version that utilizes the multi-magnification factor in WSIs. The two modules can be plugged in most existing MIL pipelines with easy code manipulation, and offer an enhanced performance. In this repo, we demonstrate how to apply the modules to the [CLAM](https://github.com/mahmoodlab/CLAM) pipeline, for the classification task over the [PANDA](https://panda.grand-challenge.org/data/) challenge dataset*

<div style="width: 100%; text-align: center;">
    <img src="PMIL.gif" width="600px" />
</div>

[Installation](#installation) • [Patchifying and Feature Extraction](#patchifying-and-feature-extraction) • [Nuclei Extraction](#nuclei-extraction) • [Topological Data Extraction](#topological-data-extraction) • [Codes Modification](#codes-modification) • [TDA Features Extraction](#tda-features-extraction) • [Checkpoints](#checkpoints) • [Cite](#reference)




## Installation:
Clone this repo, and clone [CLAM](https://github.com/mahmoodlab/CLAM) repo. The general recommendation is to create two environments, one for the topology analysis, and one for the CLAM pipeline.
Follow installation instructions of [CLAM](https://github.com/mahmoodlab/CLAM) and [Topology Repo](github.com/KitwareMedical/HistologyCancerDiagnosisDeepPersistenceHomology).

## Patchifying and Feature Extraction
Refer to the [CLAM](https://github.com/mahmoodlab/CLAM) for details about patchifying, and RGB feature extraction. We provide the csv files that we used to extract the patches in PANDA.csv.

## Nuclei Extraction
It is important to extract nuclei, as they represent the main key features in the histopathology image. The point cloud, represented by the extracted nuclei, will be the catalyst to start our TDA. Nuclei extraction can either be done using simple thresholding, or using a pretrained model like the [hovernet](https://github.com/vqdang/hover_net) model. In this example, we used the former.
If a pretrained model is used, save the nuceli detection results in a json file (example included: nuc_example.json). This must be done for each patch in the WSI at a time.

We supply a profiling pipeline that facilitates an improved nuclei extraction based on thresholding and color convolution in displays/. By manual inspection, we extracted the minimum and maximum RGB values for each class in the PANDA dataset. For each given WSI, nuclei can be extracted based on these RGB values. This is done automatically in the code, so no need to worry about it. 
To opt out of this and follow the origianl method, you can follow [this](https://github.com/KitwareMedical/HistologyCancerDiagnosisDeepPersistenceHomology).


## Topological Data Extraction
For the topology analysis we follow the nice work of [Kitware](https://github.com/KitwareMedical/HistologyCancerDiagnosisDeepPersistenceHomology).
To extract the Persistance Images:
```
python gen_pers.py --data <dir to data> \
                   --type_handle <"Gleason grade (0,1,2,3,4,5)"> \
                   --magn <magnification factor (16.0, 4.0, 1.0)> \
                   --split <train/test/va split> \
                   --extracted_nucs <1 if using a pretrained model, otherwise 0>
```
Afterwards, PI images will be saved to the output directory and used in the subsequent steps.

## Codes Modification
The following lists all the modified files and the changes they undertook. 

*NOTE:* The modified files can all be found in CLAM__mod/. Simply replace the existing files in the original CLAM repo by the modified ones.


To apply these changes to a different MIL pipeline, use the same logic, which primarily revolves around modifying:
1. Feature extractors (for the PC case) (1 to 3)
2. Data Loaders (4, 5)
3. Main model (6 to 9)
### 1. extract_features.py
1. Include two new model names in the --model_name argument. Namely, 'resnet50_trunc_3d', 'resnet50_trunc_PI', which extract PC and PI features.
2. Include a modality argument to separate PIs pipelines from PCs ones.
### 2. models/builder.py
The function get_encoder() used inside extract_features.py is modified to facilitate the usage of 'resnet50_trunc_3d', 'resnet50_trunc_PI' models.
### 3. models/timm_wrapper.py
The function TimmCNNEncoder is modified to use timm_3d, if needed, or continue with the 2D timm models for the RGB/PI case.
### 4. utils/constants.py
The constants used for image transformation are modified
### 5. dataset_modules/dataset_h5.py
The class Whole_Slide_Bag is modified to accomodate the loading of PIs/PCs
### 6. main.py
The main training/testing script is modified to integrate H0/H1 features, specify the magnification level of analysis, and specify the modality (vanilla, PIs, PCs).
### 7. dataset_modules/dataset_generic.py
The Generic_MIL_Dataset() function and other supporting functions are modified to accomodate the loading of PIs/PCs features.
### 8. utils/core_utils.py
The train() and function is modified to accomodate calling the right model, loading the right data into it.
### 9. models/model_clam.py
The main model CLAM_SB is modified to facilitate feature-level fusion of RGB features and PI (H0 or H1) or PC (H0 or H1) features.

## TDA Features Extraction
The TDA features are extracted in the same way the vanilla RGB features are extracted following [CLAM](https://github.com/mahmoodlab/CLAM). 

After going through the necessary code modification to facilitate handling PI images, both PI and PC features can be extracted and used in training.

To extract the PI/PC features:
```
CUDA_VISIBLE_DEVICES=1 python extract_features.py \
--data_dir <dir to PIs> \
--csv_path <dir to csv path> \
--feat_dir <output dir> \
--batch_size 512 \
--slide_ext .tiff \
--model_name "resnet50_trunc_PI" (or "resnet50_trunc_PI_3d") \
--modality "PIs" (or "PCs") 
```

## Checkpoints
For reproducability, all trained models used can be accessed [here](https://drive.google.com/drive/folders/1yW1mPhAYypg2-ZwWmRU1LetyW2wfFE8M?usp=sharing).
Make a directory called shared_weights and place the contents in the above link inside.

## Testing
The results in the paper can be reproduced using the above weights, and running main.py with the argument --test_decision, and setting --modality to PIs or PCs.

For PMIL:
```
CUDA_VISIBLE_DEVICES=1 python PMIL_main.py \
 --task task_2_tumor_subtyping \
 --model_type clam_sb \
 --data_root_dir ~/../../mnt/nvme0n1/Dataset/classification/histopathology_classification/PANDA/CLAM_256x256/train/Featurized_data_4 \
 --magn 4.0 \
 --subtyping \
 --test_decision 1 \
 --modality PIs \
```

For cPMIL, the magnification should always be the highest level:
```
CUDA_VISIBLE_DEVICES=1 python PMIL_main.py \
 --task task_2_tumor_subtyping \
 --model_type clam_sb \
 --data_root_dir ~/../../mnt/nvme0n1/Dataset/classification/histopathology_classification/PANDA/CLAM_256x256/train/Featurized_data_16 \
 --magn 16.0 \
 --subtyping \
 --test_decision 1 \
 --modality PCs \
```


| PMIL | cPMIL |
|:------:|:------:|
| ACC=0.731 | ACC=0.663 |
| AUC=0.936| AUC=0.909 |


## Issues
- Please report all issues on the public forum.




## Reference
Coming Soon!
