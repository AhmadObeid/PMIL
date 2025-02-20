PMIL <img src="pmil-logo.png" width="280px" align="right" />
===========

***Overview:** PMIL is a versatile module for modeling object-level information in histopathology images through Topological Data Analysis. cPMIL is a cubic version that utilizes the multi-magnification factor in WSIs. In this repo, we apply both modules to ...etc*

<img src="model.gif" width="470px" align="center" />


[Pre-requisites](#pre-requisites) • [Installation](INSTALLATION.md) • [Patchifying and Feature Extraction](#wsi-segmentation-and-patching) • [Nuclei Extraction](#Training-Splits) • [Topological Data Extraction](#topological-data-extraction) • [TDA-Features](#TDA-features) • [Checkpoints](#Checkpoints) • [Codes Modification](#codes-modification) • [Examples](#examples) • [Cite](#reference)



## Updates:
* **01/03/2025**: Our cPMIL is submitted for revision.
* **02/08/2025**: Our PMIL model has been accepted at ISCAS 2025.


## Installation:
Please refer to our [Installation guide](INSTALLATION.md) for detailed instructions on how to get started.

## Patchifying and Feature Extraction
Refer to the [CLAM](www.github.com/CLAM) for details about patchifying, and RGB feature extraction. We provide the csvs for reference.

## Nuclei Extraction
It is important to extract nuclei, as they represent the main key features in the histopathology image. The point cloud, represented by the extracted nuclei, will be the catalyst to start our TDA. Nuclei extraction can either be done using simple thresholding, or using a pretrained model like the [hovernet](www.github.com/hovernet) model. In this example, we used the former.


## Topological Data Extraction
If nuclei are extracted before strating TDA (i.e. through a pretrained model), they must be passed as argument.
otherwise...etc

## TDA-features


### Checkpoints
For reproducability, all trained models used can be accessed [here](https://drive.google.com/drive/folders/1NZ82z0U_cexP6zkx1mRk-QeJyKWk4Q7z?usp=sharing).


## Codes Modification


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
