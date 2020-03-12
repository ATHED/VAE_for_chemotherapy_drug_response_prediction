

library(ggplot2)
#########################################
#calculate 95% confidence interval
##########################################
#expression data
auroc.data <- read.table("/media/qiwei/work/Python_playground/VAE/TCGA_5_cancers/auroc_binary_labels/Comparison_vae_pca_5fold(SARC)(5cancers_best).csv"
                                       , sep=",",
                                       header=T, stringsAsFactors=FALSE, row.names = 1,
                                       quote="",
                                       comment.char="#")
colnames(auroc.data)

#auroc.data <- auroc.data[,-which(colnames(auroc.data) %in% c("pca.whether.selected.by.xgb", "grade.whether.selected.by.xgb", "stage.whether.selected.by.xgb" ))]


#temp <- stack(auroc.data)
#####################
#boxplot
###########
#generate the graph
ggplot(stack(auroc.data), aes(x = ind, y = values, color = ind)) + 
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(shape=16, position=position_jitter(0.2)) +
  labs(title = "Comparison of AUROC, top 20% high variance genes, with/not VAE", x = "Method", y = "Auroc") +
  theme(axis.text = element_text(size = 12), axis.title=element_text(size=14,face="bold")) +
  scale_y_continuous(breaks = seq(min(auroc.data), max(auroc.data), by = 0.02) ) +
  ggsave("fig_top20_vae_ref_replications(SARC, 5_cancers, 200220).pdf", width=15, height=9,limitsize = FALSE)


#########################################################
#calculate statistic significance using paired t-test
#######################################################
colnames(auroc.data)

#pca_0.9_1.0_t.test <- t.test(x = auroc.data$PCA_compressed_og_n359, y = auroc.data$PCA_compressed_og_n1217, paired = TRUE, alternative = "two.sided")
#pca_0.9_1.0_t.test

og_vae_t.test <- t.test(x = auroc.data$PerS_vae_NN6k3L_z500_minmax, y = auroc.data$Original_dataset_n12097, paired = TRUE, alternative = "two.sided")
og_vae_t.test

pca_vae_t.test <- t.test(x = auroc.data$PerS_vae_NN6k3L_z500_minmax, y = auroc.data$PCA_compressed_og_n176, paired = TRUE, alternative = "two.sided")
pca_vae_t.test

pca_vae_w.test <- wilcox.test(x = auroc.data$PerS_vae_NN6k3L_z500_minmax, y = auroc.data$PCA_compressed_og_n176, paired=TRUE, alternative = "two.sided") 
pca_vae_w.test

#calculate the 95% confidence interval
summary(auroc.data$X3Layers_0.1test_high_var_0.2)
length(auroc.data$X3Layers_0.1test_high_var_0.2)
mean(auroc.data$X3Layers_0.1test_high_var_0.2)
sd(auroc.data$X3Layers_0.1test_high_var_0.2)

error <- qt(0.975,df=length(auroc.data$X3Layers_0.1test_high_var_0.2)-1)*sd(auroc.data$X3Layers_0.1test_high_var_0.2)/sqrt(length(auroc.data$X3Layers_0.1test_high_var_0.2))
error
left <- mean(auroc.data$X3Layers_0.1test_high_var_0.2)-error
right <- mean(auroc.data$X3Layers_0.1test_high_var_0.2)+error

print(paste("the 95% confidence interval based on 10 replications is: [",left,", ",right,"]"))

####################################
#clean
#######################
#rm(list=ls(all=TRUE))