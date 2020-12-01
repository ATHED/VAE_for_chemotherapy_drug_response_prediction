# VAE_for_chemotherapy_drug_response_prediction

This a program in progress. The idea is using variational auto-encoder to extract lower dimension abstract data from gene expression data. Then applying those lower dimension abstract data to predict chemotherapy response on various type of cancers.

A preprint based on this research will be released soon.

You can create the virtual environment contains all necessory packages from the enviroment file provided, and with code:
```
conda env create -f tensorflow-gpu-environment-stable.yml

```

You need to install Cuda, and Cudnn on you workstation first.

## Diagrammatic representation of the VAE-XGBoost method that we used for predicting tumor response.
![Image of our VAE model](https://github.com/ATHED/VAE_for_chemotherapy_drug_response_prediction/blob/master/images/m1_pipeline_plot_modified.png)

Input layer with top 20 most variably expressed genes (size *m*), configurable multiple fully connected dense layers (e.g., three layers, and six layers) as encoding neural network (encoder), encoder outputting two vectors of configurable latent variable size *h* (*h*: manually selected latent vector size, e.g, $50$, $500$, $1000$, etc., *h* << *m* for dimension reduction): a vector of means &mu, and another vector of standard deviations &sigma. They form the parameters of a vector of random variables of length *h*, with *i*th element of &mu<sub>i and &sigma<sub>i being the mean and standard deviation of the *i*th random variable **z**<sub>i. The sampled encoding then being passed to the decoding neural network (decoder), which is configured by the same number of fully connected dense layers as the encoder part. The decoder will then reconstruct the training data. 
