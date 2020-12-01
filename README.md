# VAE_for_chemotherapy_drug_response_prediction

The idea is using a variational auto-encoder to extract lower dimension abstract data from gene expression data. Then applying those lower dimension abstract data to predict chemotherapy response on various type of cancers using xgboost classifier.

A preprint based on this research will be released soon. Here is the link:

You can create the virtual environment contains all necessory packages from the enviroment file provided, and with code:
```
conda env create -f tensorflow-gpu-environment-stable.yml

```

You need to install Cuda, and Cudnn on you workstation first.
