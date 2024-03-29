# VAE for chemotherapy drug response prediction

Code repository accompanying the published paper *"Predicting chemotherapy response using a variational autoencoder approach"* by Qi Wei and Stephen A. Ramsey *BMC Bioinformatics (2021) 22:453*.

You can find full text of my paper on this link: https://doi.org/10.1186/s12859-021-04339-6

Your can cite my paper using the following bibtex citation:
```
@article{article,
author = {Wei, Qi and Ramsey, Stephen},
year = {2021},
month = {09},
pages = {453},
title = {Predicting chemotherapy response using a variational autoencoder approach},
volume = {22},
journal = {BMC Bioinformatics},
doi = {10.1186/s12859-021-04339-6}
}
```

## Introduction & Installation

The idea is using a variational auto-encoder to extract lower dimension abstract data from gene expression data. Then applying those lower dimension abstract data to predict chemotherapy response on various type of cancers using xgboost classifier.

You can create the virtual environment contains all necessory packages from the enviroment file provided, and with code:
```
conda env create -f tensorflow-gpu-environment-stable.yml
```

Notice: You need to install Cuda, and Cudnn on you workstation first. You may find instructions here: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

## Training Dataset
* Dataset link:https://xenabrowser.net/datapages/
* Clinical record link:https://xenabrowser.net/datapages/
                       https://www.cbioportal.org/datasets
* Five cancer types: colon adenocarcinomas (COAD),
pancreatic adenocarcinoma (PAAD), bladder carcinoma (BLCA), sarcoma (SARC), and breast invasive carcinoma (BRCA), selected based on availability of a sufficient amount of labeled data in TCGA
* Binary labels for chemotherapy response constructed from the combined clinical record: https://github.com/ATHED/VAE_for_chemotherapy_drug_response_prediction/tree/master/binary_labels 

## Code Structure
* VAE training code: https://github.com/ATHED/VAE_for_chemotherapy_drug_response_prediction/tree/master/VAE_models
* R script for adding chemotherapy treatment response labels: https://github.com/ATHED/VAE_for_chemotherapy_drug_response_prediction/tree/master/R_scripts_for_adding_labels
* Classification task codes for VAE, PCA, and original dimension data: https://github.com/ATHED/VAE_for_chemotherapy_drug_response_prediction/tree/master/Benchmark_codes_xgboost
* Plotting scripts for results: https://github.com/ATHED/VAE_for_chemotherapy_drug_response_prediction/tree/master/R_scripts_for_plotting

Notebook version of codes: https://github.com/ATHED/VAE_for_chemotherapy_drug_response_prediction/tree/master/Notebooks

## Diagrammatic representation of the VAE-XGBoost method that we used for predicting tumor response.
![Image of our VAE model](https://github.com/ATHED/VAE_for_chemotherapy_drug_response_prediction/blob/master/images/m1_pipeline_plot_modified.png)

Input layer with top 20 most variably expressed genes (size *m*), configurable multiple fully connected dense layers (e.g., three layers, and six layers) as encoding neural network (encoder), encoder outputting two vectors of configurable latent variable size *h* (*h*: manually selected latent vector size, e.g, 50, 500, 1000, etc., *h* << *m* for dimension reduction): a vector of means <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mu" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\mu" title="\mu" /></a>, and another vector of standard deviations <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\sigma" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\sigma" title="\sigma" /></a>. They form the parameters of a vector of random variables of length *h*, with *i*th element of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mu&space;_i" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\mu&space;_i" title="\mu _i" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\sigma&space;_i" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\sigma&space;_i" title="\sigma _i" /></a> being the mean and standard deviation of the *i*th random variable <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\boldsymbol{\mathrm{z}}&space;_i" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\boldsymbol{\mathrm{z}}&space;_i" title="\boldsymbol{\mathrm{z}} _i" /></a>. The sampled encoding then being passed to the decoding neural network (decoder), which is configured by the same number of fully connected dense layers as the encoder part. The decoder will then reconstruct the training data.

## Results
![Image of Table 1](https://github.com/ATHED/VAE_for_chemotherapy_drug_response_prediction/blob/master/images/VAE-paper-Table1.png)

![Image of Table 1](https://github.com/ATHED/VAE_for_chemotherapy_drug_response_prediction/blob/master/images/VAE-paper-Table2.png)

![Image of Table 1](https://github.com/ATHED/VAE_for_chemotherapy_drug_response_prediction/blob/master/images/VAE-paper-Table3.png)

