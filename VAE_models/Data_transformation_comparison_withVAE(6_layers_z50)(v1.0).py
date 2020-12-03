
# coding: utf-8

# # Load packages and use tensorflow as backend

# In[1]:


# Install a pip package in the current Jupyter kernel
import sys
#!{sys.executable} -m pip install numpy
#!{sys.executable} -m pip install requests
#import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
import tensorflow as tf
import keras
from tensorflow.python.client import device_lib
import os
import seaborn as sns

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics, optimizers
from keras import losses
from keras.callbacks import Callback
import keras
from keras import utils

import pydot
import graphviz
from keras.utils import plot_model
from keras_tqdm import TQDMNotebookCallback
#from .tqdm_callback import TQDMNotebookCallback
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[2]:


#test tensorflow, remember to change the kernel
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(10)
b = tf.constant(32)
print("10 + 32 = ", sess.run(a + b))


# # Check the system information

# In[3]:


#######################################################################################################
#check the system information, check if cuda and gpu computing for tensorflow is installed properly
######################################################################################################
print("whether tensorflow is built with cuda: ", tf.test.is_built_with_cuda())
print("whether gpu computing is available for tensorflow: ", tf.test.is_gpu_available())
print("using keras version: ", keras.__version__)
print("using tensorflow version: ", tf.__version__)
print("\n")
print("Device details:\n", device_lib.list_local_devices())
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


# # Reading files/documents

# In[4]:


#Reading files/documents
#using all unlabeled data
path = '/media/qiwei/work/Python_playground/VAE/TCGA_5_cancers/counts_data/high_var_counts_data/TCGA_(BLCA_COAD_SARC_PAAD_BRCA)_(0.2chemo)VSTnrom_count_expr_clinical_data.txt'

#only use labeled data
#path = "counts_data/counts_data_without_label/TCGA_SARC_(0.2chemo_45samples)VSTnrom_count_expr_clinical_data.tsv"

#open(path).readline()
#gene expression RNAseq, Batch effects normalized mRNA data

ExprAlldata = pd.read_csv(path, sep = "\t", index_col = 0)
ExprAlldata = ExprAlldata.dropna(axis='columns')
#ExprAlldata.columns = ["Gene", "Counts"]
print("The dimension of input dataset is: ", ExprAlldata.shape)


# In[5]:


ExprAlldata.head(3)


# ## Sanity check

# In[6]:


any_na = np.any(np.isnan(ExprAlldata))
print ('There exists NA value: ' + repr (any_na))

all_finite = np.all(np.isfinite(ExprAlldata))
print ('All values are finite: ' + repr (all_finite))


# # Data normalization choices

# In[7]:


#minmax data transformation
from sklearn import preprocessing

#built up data frame
from pandas import DataFrame, Series
Exprframe = DataFrame(ExprAlldata)
#Exprframe = ExprAlldata.T
Exprframe_og = Exprframe

# Scale RNAseq data using zero-one normalization
Exprframe_zerone = preprocessing.MinMaxScaler().fit_transform(Exprframe)
Exprframe_zerone.shape

#change column name
#Exprframe.columns.values[0] = "Gene"

#set rownames
#Exprframe = Exprframe.set_index('Gene')
#Exprframe


# In[8]:


# logistic transformation, logistic sigmoid function
#def logits(x):
#    return 1 / (1 + np.exp(-x))

#Exprframe_logit = logits(Exprframe)
#Exprframe_logit.shape


# In[9]:


# Standardize
#scaler = preprocessing.StandardScaler()
#scaler.fit((Exprframe))
#Exprfram_std = scaler.transform(Exprframe)
#Exprfram_std.shape


# In[10]:


# If select the minmax method
Exprframe = pd.DataFrame(Exprframe_zerone,
                         columns=Exprframe.columns,
                         index=Exprframe.index)

# If select the logistic transformation method
#Exprframe = pd.DataFrame(Exprframe_logit,
#                         columns=Exprframe.columns,
#                         index=Exprframe.index)

# If select the Standardization method
#Exprframe = pd.DataFrame(Exprfram_std,
#                         columns=Exprframe.columns,
#                         index=Exprframe.index)

# If use no transformation
#Exprframe = Exprframe_og

#print(Exprframe.shape)
#Exprframe.head(3)

#output log transformed data
#log_file = "counts_data/vae_compressed/log_transformed(0.2_var,3layers,0.1test,log).tsv"
#Exprframe.to_csv(log_file, sep='\t')


# In[11]:


import math
#contruct training dataset
n_genes = Exprframe.shape[1]
print ('number of genes is ' + repr (n_genes))


# ## Split 10% of the data as test set randomly

# In[13]:


#import the data as training data
#set the random state to 42

# Split 10% test set randomly
test_set_percent = 0.1
Exprframe_test = Exprframe.sample(frac=test_set_percent, random_state = 42)
Exprframe_train = Exprframe.drop(Exprframe_test.index)
print("The dimension of training dataset is: ",Exprframe_train.shape)


# # Load functions and classes
# * This will facilitate connections between layers and also custom hyperparameters

# In[14]:


# Function for reparameterization trick to make model differentiable
def sampling(args):
    
    import tensorflow as tf
    # Function with args required for Keras Lambda function
    z_mean, z_log_var = args

    # Draw epsilon of the same shape from a standard normal distribution
    epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                              stddev=epsilon_std)
    
    # The latent vector is non-deterministic and differentiable
    # in respect to z_mean and z_log_var
    z = z_mean + K.exp(z_log_var / 2) * epsilon
    return z


class CustomVariationalLayer(Layer):
    """
    Define a custom layer that learns and performs the training
    This function is borrowed from:
    https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
    """
    def __init__(self, **kwargs):
        # https://keras.io/layers/writing-your-own-keras-layers/
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    #def vae_loss(self, x_input, x_decoded):
    #    reconstruction_loss = original_dim * metrics.binary_crossentropy(x_input, x_decoded)
    #    kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) - 
    #                            K.exp(z_log_var_encoded), axis=-1)
    #    return K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))
    
    def vae_loss(self, x_input, x_decoded):
        #per sample
        reconstruction_loss = original_dim * losses.mean_absolute_error(x_input, x_decoded)
        kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) - 
                                K.exp(z_log_var_encoded), axis=-1)
        
        #
        #per data point
        #reconstruction_loss = losses.mean_absolute_error(x_input, x_decoded)
        #kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) - 
        #                        K.exp(z_log_var_encoded), axis=-1) / latent_dim
        
        
        return K.mean(reconstruction_loss + alpha * (kl_loss))#K.mean(reconstruction_loss + (K.get_value(beta) * kl_loss))


    def call(self, inputs):
        x = inputs[0]
        x_decoded = inputs[1]
        loss = self.vae_loss(x, x_decoded)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x


# ## Implementing Warm-up as described in Sonderby et al. LVAE
# 
# * This is modified code from https://github.com/fchollet/keras/issues/2595

# In[15]:


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa
    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)


# ## Set hyper parameters

# In[16]:


# Set hyper parameters
original_dim = Exprframe.shape[1]
print("The dimension of input layer is: ", original_dim)

layer1_dim = 6000
layer2_dim = 3000
layer3_dim = 1000
layer4_dim = 500
layer5_dim = 100
latent_dim = 50

batch_size = Exprframe.shape[0]
epochs = 400
learning_rate = 0.002

#set kernel initializer
# Casey paper 'glorot_uniform'
#initial_method = 'glorot_uniform'
#initial_method = keras.initializers.glorot_uniform(seed=807)

initial_method = keras.initializers.glorot_normal(seed=42)

epsilon_std = 1.0
alpha = 1.0

beta = K.variable(0)
kappa = 0.002


# # Encoder network

# In[17]:


#simple neural network version with two layers
#Layer 1
# Input place holder for RNAseq data with specific input size
rnaseq_input = Input(shape=(original_dim, ))

#L1
l1_dense_linear = Dense(layer1_dim, kernel_initializer=initial_method)(rnaseq_input)
l1_dense_batchnorm = BatchNormalization()(l1_dense_linear)
l1 = Activation('relu')(l1_dense_batchnorm)

#l2
l2_dense_linear = Dense(layer2_dim, kernel_initializer=initial_method)(l1)
l2_dense_batchnorm = BatchNormalization()(l2_dense_linear)
l2 = Activation('relu')(l2_dense_batchnorm)

#l3
l3_dense_linear = Dense(layer3_dim, kernel_initializer=initial_method)(l2)
l3_dense_batchnorm = BatchNormalization()(l3_dense_linear)
l3 = Activation('relu')(l3_dense_batchnorm)

#l4
l4_dense_linear = Dense(layer4_dim, kernel_initializer=initial_method)(l3)
l4_dense_batchnorm = BatchNormalization()(l4_dense_linear)
l4 = Activation('relu')(l4_dense_batchnorm)

#l5
l5_dense_linear = Dense(layer5_dim, kernel_initializer=initial_method)(l4)
l5_dense_batchnorm = BatchNormalization()(l5_dense_linear)
l5 = Activation('relu')(l5_dense_batchnorm)


# In[20]:


#Layer 6
# Input layer is compressed into a mean and log variance vector of size `latent_dim`
# Each layer is initialized with glorot uniform weights and each step (dense connections,
# batch norm, and relu activation) are funneled separately
# Each vector of length `latent_dim` are connected to the rnaseq input tensor

z_mean_dense_linear = Dense(latent_dim, kernel_initializer=initial_method)(l5)
z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

z_log_var_dense_linear = Dense(latent_dim, kernel_initializer=initial_method)(l5)
z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

# return the encoded and randomly sampled z vector
# Takes two keras layers as input to the custom sampling function layer with a `latent_dim` output
z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean_encoded, z_log_var_encoded])


# # Decoder Network

# In[21]:


# The decoding layers have 6 layers and relu activation
decoderl5_reconstruct = Dense(layer5_dim, kernel_initializer=initial_method, activation='relu')
decoder_l5 = decoderl5_reconstruct(z)

decoderl4_reconstruct = Dense(layer4_dim, kernel_initializer=initial_method, activation='relu')
decoder_l4 = decoderl4_reconstruct(decoder_l5)

decoderl3_reconstruct = Dense(layer3_dim, kernel_initializer=initial_method, activation='relu')
decoder_l3 = decoderl3_reconstruct(decoder_l4)

decoderl2_reconstruct = Dense(layer2_dim, kernel_initializer=initial_method, activation='relu')
decoder_l2 = decoderl2_reconstruct(decoder_l3)

decoderl1_reconstruct = Dense(layer1_dim, kernel_initializer=initial_method, activation='relu')
decoder_l1 = decoderl1_reconstruct(decoder_l2)

decoderl0_reconstruct = Dense(original_dim, kernel_initializer=initial_method, activation='relu')
rnaseq_reconstruct = decoderl0_reconstruct(decoder_l1)


# ## Connect the encoder and decoder to make the VAE
# 
# * The CustomVariationalLayer() includes the VAE loss function (reconstruction + (beta * KL)), which is what will drive our model to learn an interpretable representation of gene expression space.
# 
# * The VAE is compiled with an Adam optimizer and built-in custom loss function. The loss_weights parameter ensures beta is updated at each epoch end callback

# In[23]:


from keras import losses
adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
vae_layer = CustomVariationalLayer()([rnaseq_input, rnaseq_reconstruct])
vae = Model(rnaseq_input, vae_layer)
vae.compile(optimizer=adam, loss=None, loss_weights=[beta])
#vae.compile(optimizer=adam, loss=losses.kullback_leibler_divergence)

#########################################################################
#only use to manually set initial weights, otherwise change the initializer
weights = vae.get_weights()
#new_weight = [item*0+0.01 for item in weights]
#vae.set_weights(new_weight)
vae.summary()


# # Train the model
# 
# * The training data is shuffled after every epoch and 10% of the data is heldout for calculating validation loss.

# In[24]:


get_ipython().run_cell_magic('time', '', 'hist = vae.fit(np.array(Exprframe_train),\n               shuffle=True,\n               epochs=epochs,\n               verbose=0,\n               batch_size=batch_size,\n               validation_data=(np.array(Exprframe_test), None),\n               callbacks=[WarmUpCallback(beta, kappa),\n                          TQDMNotebookCallback(leave_inner=True, leave_outer=True)])')


# In[26]:


#hist.history
z5000_df = pd.DataFrame(hist.history)
z5000_df.loc[399]


# In[27]:


# Visualize training performance
history_df = pd.DataFrame(hist.history)
history_df = history_df.iloc[60:399]

hist_plot_file = "temp.pdf"#"(Lr_0.002)(NN6K_z100_a1.0_6L_0.1t)obj_func_per_dp(4cancers).pdf"#"temp.pdf"#
ax = history_df.plot()

ratio = 0.95
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
# the abs method is used to make sure that all numbers are positive
# because x and y axis of an axes maybe inversed.
ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

ax.set_xlabel('Epochs')
ax.set_ylabel('objective function (include both reconstruct_loss, kl_loss)')
ax.set_title('')
fig = ax.get_figure()
#fig.savefig(hist_plot_file)


# # Extract Encoder model

# In[28]:


#extract the encoder part

# Model to compress input
#encoder = Model(rnaseq_input, [z_mean_encoded, z_log_var_encoded])
encoder = Model(rnaseq_input, z)
encoder.summary()


# In[29]:


encoder2 = Model(rnaseq_input, [z_mean_encoded, z_log_var_encoded])
encoder2.summary()


# In[30]:


# Encode rnaseq into the hidden/latent representation - and save output
z_df = encoder.predict_on_batch(Exprframe_test)

z_df = pd.DataFrame(z_df, index=Exprframe_test.index)

z_df.columns.name = 'sample_id'
z_df.columns = z_df.columns + 1
z_df.head(10)

[z_mean_d, z_log_var_d]= encoder2.predict_on_batch(Exprframe_test)
z_mean_df = pd.DataFrame(z_mean_d, index=Exprframe_test.index)

z_mean_df.columns.name = 'sample_id'
z_mean_df.columns = z_mean_df.columns + 1


z_log_var_df = pd.DataFrame(z_log_var_d, index=Exprframe_test.index)

z_log_var_df.columns.name = 'sample_id'
z_log_var_df.columns = z_log_var_df.columns + 1


# # Extract Decoder model

# In[31]:


# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim, ))  # can generate from any sampled z vector

_x_decoded_l5 = decoderl5_reconstruct(decoder_input)
_x_decoded_l4 = decoderl4_reconstruct(_x_decoded_l5)
_x_decoded_l3 = decoderl3_reconstruct(_x_decoded_l4)

_x_decoded_l2 = decoderl2_reconstruct(_x_decoded_l3)

_x_decoded_l1 = decoderl1_reconstruct(_x_decoded_l2)
_x_decoded_l0 = decoderl0_reconstruct(_x_decoded_l1)

decoder = Model(decoder_input, _x_decoded_l0)
decoder.summary()


# ## Observe reconstruction fidelity

# In[35]:


#original input RNAseq data
rnaseq_df = Exprframe_test
rnaseq_df.head(3)


# In[36]:


# How well does the model reconstruct the input RNAseq data
input_rnaseq_reconstruct = decoder.predict(np.array(z_df))
input_rnaseq_reconstruct = pd.DataFrame(input_rnaseq_reconstruct, index=rnaseq_df.index,
                                        columns=rnaseq_df.columns)
input_rnaseq_reconstruct.head(3)


# In[37]:


#test the fidelity
reconstruction_fidelity = abs(rnaseq_df - input_rnaseq_reconstruct)

reconstruction_loss = reconstruction_fidelity.mean(axis = 1)

#print(reconstruction_loss)

gene_mean = reconstruction_fidelity.mean(axis=0)
gene_abssum = reconstruction_fidelity.abs().sum(axis=0).divide(rnaseq_df.shape[0])
gene_summary = pd.DataFrame([gene_mean, gene_abssum], index=['gene mean', 'gene abs(sum)']).T
#gene_summary.sort_values(by='gene abs(sum)', ascending=False).head(20)


# # Print out the mean reconstruction loss and the mean KL loss

# In[43]:


# L1 loss: losses.mean_absolute_error
reconstruction_loss_used = losses.mean_absolute_error(rnaseq_df, input_rnaseq_reconstruct) #* original_dim
with tf.Session() as sess:
    #print the reconstruction loss that we calculated
    mean_reconstruct_loss = sess.run(K.mean(reconstruction_loss_used))
    print ("The mean reconstruction loss for each data point is: %.11f" % mean_reconstruct_loss)


# In[44]:


kl_loss = - 0.5 * K.sum(1 + z_log_var_d - K.square(z_mean_d) - 
                                K.exp(z_log_var_d), axis=-1) / latent_dim
with tf.Session() as sess:
    #print the kl loss that we calculated
    mean_kl_loss = sess.run(K.mean(kl_loss))
    print ("The mean KL loss for each data point is: %.11f" %mean_kl_loss)


# In[46]:


print ("The combined mean loss for each data point is: %.11f" % (mean_reconstruct_loss + alpha * mean_kl_loss))
print ("The current alpha choice for reconstruction loss + alpha*kl loss is: ", alpha)


# In[ ]:


# Encode rnaseq into the hidden/latent representation - and save output
#encoded_rnaseq_df

z_df = encoder.predict_on_batch(Exprframe)

z_df = pd.DataFrame(z_df, index=Exprframe.index)

z_df.columns.name = 'sample_id'
z_df.columns = z_df.columns + 1
z_df.head(10)

encoded_file = "counts_data/vae_compressed/encoded_5cancers_rnaseq_vae(perSp,a1.0,unlabel,0.2_var,6LF6k,z50,minmax,ep700).tsv"
#encoded_file = "counts_data/vae_compressed/encoded_4_cancers_rnaseq_vae(perSp,a0,unlabel,0.2_var,3LF6k,z500,minmax).tsv"
z_df.to_csv(encoded_file, sep='\t')

