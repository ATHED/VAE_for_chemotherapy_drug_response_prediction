library(ggplot2)
library(reshape2)
#########################################
#calculate 95% confidence interval
##########################################
#expression data
#auroc.data <- read.table("/media/qiwei/work/Python_playground/VAE/TCGA_5_cancers/auroc_binary_labels/Comparison_vae_pca_5fold(SARC)(5cancers, preprint).csv"
#                                       , sep=",",
#                                       header=T, stringsAsFactors=FALSE, row.names = 1,
#                                       quote="",
#                                       comment.char="#")

#SARC
auroc.SARC.data <- read.table("Comparison_vae_pca_5fold(SARC)(5cancers, preprint).csv"
                         , sep=",",
                         header=T, stringsAsFactors=FALSE, row.names = 1,
                         quote="",
                         comment.char="#")

colnames(auroc.SARC.data)
colnames(auroc.SARC.data) <- c("Raw(12,097)", "PCA(387)", "VAE(500)")
auroc.SARC.stack <- stack(auroc.SARC.data)
auroc.SARC.stack$cancer_type <- "SARC"

#BLCA
auroc.BLCA.data <- read.table("Comparison_vae_pca_5fold(BLCA)(5cancers, preprint).csv"
                              , sep=",",
                              header=T, stringsAsFactors=FALSE, row.names = 1,
                              quote="",
                              comment.char="#")

colnames(auroc.BLCA.data)
colnames(auroc.BLCA.data) <- c("Raw(12,097)", "PCA(387)", "VAE(50)")
auroc.BLCA.stack <- stack(auroc.BLCA.data)
auroc.BLCA.stack$cancer_type <- "BLCA"

#BRCA
auroc.BRCA.data <- read.table("Comparison_vae_pca_5fold(BRCA)(5cancers, preprint).csv"
                              , sep=",",
                              header=T, stringsAsFactors=FALSE, row.names = 1,
                              quote="",
                              comment.char="#")

colnames(auroc.BRCA.data)
colnames(auroc.BRCA.data) <- c("Raw(12,097)", "PCA(387)", "VAE(50)")
auroc.BRCA.stack <- stack(auroc.BRCA.data)
auroc.BRCA.stack$cancer_type <- "BRCA"

#PAAD
auroc.PAAD.data <- read.table("Comparison_vae_pca_5fold(PAAD)(5cancers, preprint).csv"
                              , sep=",",
                              header=T, stringsAsFactors=FALSE, row.names = 1,
                              quote="",
                              comment.char="#")

colnames(auroc.PAAD.data)
colnames(auroc.PAAD.data) <- c("Raw(12,097)", "PCA(387)", "VAE(50)")
auroc.PAAD.stack <- stack(auroc.PAAD.data)
auroc.PAAD.stack$cancer_type <- "PAAD"

#COAD
auroc.COAD.data <- read.table("Comparison_vae_pca_5fold(COAD)(5cancers, preprint).csv"
                              , sep=",",
                              header=T, stringsAsFactors=FALSE, row.names = 1,
                              quote="",
                              comment.char="#")

colnames(auroc.COAD.data)
colnames(auroc.COAD.data) <- c("Raw(12,097)", "PCA(387)", "VAE(650)")
auroc.COAD.stack <- stack(auroc.COAD.data)
auroc.COAD.stack$cancer_type <- "COAD"


#rbind all the stack data frame
auroc.bind.stack <- rbind(auroc.SARC.stack, auroc.BLCA.stack)
auroc.bind.stack <- rbind(auroc.bind.stack, auroc.BRCA.stack)
auroc.bind.stack <- rbind(auroc.bind.stack, auroc.PAAD.stack)
auroc.bind.stack <- rbind(auroc.bind.stack, auroc.COAD.stack)

#clean up NA in stack
auroc.bind.clean.stack <- auroc.bind.stack[-which(auroc.bind.stack$values %in% NA),] 
auroc.bind.stack <- auroc.bind.clean.stack

#temp <- stack(auroc.data)
#####################
#boxplot
###########
#generate the graph
#ggplot(auroc.bind.stack, aes(x = ind, y = values, color = ind)) + 
#     geom_boxplot(outlier.shape = NA) +
#     geom_jitter(shape=16, position=position_jitter(0.2)) +
#     labs(y = "AUROC") +
#     theme(axis.text = element_text(size = 24), axis.title=element_text(size=36,face="bold"), strip.text.x = element_text(size = 36,face="bold"),
#          text = element_text(size =30), panel.background = element_rect(fill = "white", colour = "grey50"), legend.position = "none",
#          axis.title.x = element_blank()) +
#     scale_y_continuous(breaks = seq(floor(min(auroc.bind.stack$values)), ceiling(max(auroc.bind.stack$values)), by = 0.02) ) +
#     facet_wrap(~ cancer_type, scale="free_x", ncol = 2) +
#     ggsave("(test)fig_top20_vae_ref_replications(5_cancers, preprint).pdf", width=25, height=24,limitsize = FALSE)

##################################################
#Logit transform function
#Inverse logit (Logistic) transform function
#Logit mean function
#################################################
logit <- function(p) { log(p/(1-p)) }

Inve_logit <- function(x) {
  1.0/(1.0 + exp(-x))
}

Logit_mean <- function(x) {
  Inve_logit(mean(logit(x), na.rm = T))
}

#########################################################
#calculate statistic significance using paired t-test
#######################################################

#For BLCA
colnames(auroc.BLCA.data)

og_vae_t.BLCA.test <- t.test(x = logit(auroc.BLCA.data$`VAE(50)`), y = logit(auroc.BLCA.data$`Raw(12,097)`), paired = TRUE, alternative = "two.sided")
og_vae_t.BLCA.test

og_vae_w.BLCA.test <- wilcox.test(x= logit(auroc.BLCA.data$`VAE(50)`), y = logit(jitter(auroc.BLCA.data$`Raw(12,097)`)), paired = T, alternative = "two.sided")
og_vae_w.BLCA.test

pca_vae_t.BLCA.test <- t.test(x = logit(auroc.BLCA.data$`VAE(50)`), y = logit(auroc.BLCA.data$`PCA(387)`), paired = TRUE, alternative = "two.sided")
pca_vae_t.BLCA.test

pca_vae_w.BLCA.test <- wilcox.test(x = logit(auroc.BLCA.data$`VAE(50)`), y = logit(jitter(auroc.BLCA.data$`PCA(387)`)), paired=TRUE, alternative = "two.sided") 
pca_vae_w.BLCA.test

#For BRCA
colnames(auroc.BRCA.data)

og_vae_t.BRCA.test <- t.test(x = logit(auroc.BRCA.data$`VAE(50)`), y = logit(auroc.BRCA.data$`Raw(12,097)`), paired = TRUE, alternative = "two.sided")
og_vae_t.BRCA.test

og_vae_w.BRCA.test <- wilcox.test(x= logit(auroc.BRCA.data$`VAE(50)`), y = logit(jitter(auroc.BRCA.data$`Raw(12,097)`)), paired = T, alternative = "two.sided")
og_vae_w.BRCA.test

pca_vae_t.BRCA.test <- t.test(x = logit(auroc.BRCA.data$`VAE(50)`), y = logit(auroc.BRCA.data$`PCA(387)`), paired = TRUE, alternative = "two.sided")
pca_vae_t.BRCA.test

pca_vae_w.BRCA.test <- wilcox.test(x = logit(auroc.BRCA.data$`VAE(50)`), y = logit(jitter(auroc.BRCA.data$`PCA(387)`)), paired=TRUE, alternative = "two.sided") 
pca_vae_w.BRCA.test

#For COAD
colnames(auroc.COAD.data)

og_vae_t.COAD.test <- t.test(x = logit(auroc.COAD.data$`VAE(650)`), y = logit(auroc.COAD.data$`Raw(12,097)`), paired = TRUE, alternative = "two.sided")
og_vae_t.COAD.test

og_vae_w.COAD.test <- wilcox.test(x= logit(auroc.COAD.data$`VAE(650)`), y = logit(jitter(auroc.COAD.data$`Original dataset(12,097)`)), paired = T, alternative = "two.sided")
og_vae_w.COAD.test

pca_vae_t.COAD.test <- t.test(x = logit(auroc.COAD.data$`VAE(650)`), y = logit(auroc.COAD.data$`PCA(387)`), paired = TRUE, alternative = "two.sided")
pca_vae_t.COAD.test

pca_vae_w.COAD.test <- wilcox.test(x = logit(auroc.COAD.data$`VAE(650)`), y = logit(jitter(auroc.COAD.data$`PCA(387)`)), paired=TRUE, alternative = "two.sided") 
pca_vae_w.COAD.test

#For PAAD
colnames(auroc.PAAD.data)

og_vae_t.PAAD.test <- t.test(x = logit(auroc.PAAD.data$`VAE(50)`), y = logit(auroc.PAAD.data$`Raw(12,097)`), paired = TRUE, alternative = "two.sided")
og_vae_t.PAAD.test

og_vae_w.PAAD.test <- wilcox.test(x= logit(auroc.PAAD.data$`VAE(50)`), y = logit(jitter(auroc.PAAD.data$`Raw(12,097)`)), paired = T, alternative = "two.sided")
og_vae_w.PAAD.test

pca_vae_t.PAAD.test <- t.test(x = logit(auroc.PAAD.data$`VAE(50)`), y = logit(auroc.PAAD.data$`PCA(387)`), paired = TRUE, alternative = "two.sided")
pca_vae_t.PAAD.test

pca_vae_w.PAAD.test <- wilcox.test(x = logit(auroc.PAAD.data$`VAE(50)`), y = logit(jitter(auroc.PAAD.data$`PCA(387)`)), paired=TRUE, alternative = "two.sided") 
pca_vae_w.PAAD.test

#For SARC
colnames(auroc.SARC.data)

og_vae_t.SARC.test <- t.test(x = logit(auroc.SARC.data$`VAE(500)`), y = logit(auroc.SARC.data$`Raw(12,097)`), paired = TRUE, alternative = "two.sided")
og_vae_t.SARC.test

og_vae_w.SARC.test <- wilcox.test(x= logit(auroc.SARC.data$`VAE(500)`), y = logit(jitter(auroc.SARC.data$`Raw(12,097)`)), paired = T, alternative = "two.sided")
og_vae_w.SARC.test

pca_vae_t.SARC.test <- t.test(x = logit(auroc.SARC.data$`VAE(500)`), y = logit(auroc.SARC.data$`PCA(387)`), paired = TRUE, alternative = "two.sided")
pca_vae_t.SARC.test

pca_vae_w.SARC.test <- wilcox.test(x = logit(auroc.SARC.data$`VAE(500)`), y = logit(jitter(auroc.SARC.data$`PCA(387)`)), paired=TRUE, alternative = "two.sided") 
pca_vae_w.SARC.test



#######################################################################################################
#calculate the 95% confidence interval and mean by using logistic transform and logit transform
#with 1000 replications
######################################################################################################
#calculate mean for each cancer types
#For BLCA
colnames(auroc.BLCA.data)
BLCA.mean.vae <- Logit_mean(auroc.BLCA.data$`VAE(50)`)
BLCA.mean.pca <- Logit_mean(auroc.BLCA.data$`PCA(387)`)
BLCA.mean.raw <- Logit_mean(auroc.BLCA.data$`Raw(12,097)`)

print(paste0("BLCA.mean.vae is: ", round(BLCA.mean.vae, digits = 3)))
print(paste0("BLCA.mean.pca is: ", round(BLCA.mean.pca, digits = 3)))
print(paste0("BLCA.mean.raw is: ", round(BLCA.mean.raw, digits = 3)))

#For BRCA
colnames(auroc.BRCA.data)
BRCA.mean.vae <- Logit_mean(auroc.BRCA.data$`VAE(50)`)
BRCA.mean.pca <- Logit_mean(auroc.BRCA.data$`PCA(387)`)
BRCA.mean.raw <- Logit_mean(auroc.BRCA.data$`Raw(12,097)`)

print(paste0("BRCA.mean.vae is: ", round(BRCA.mean.vae, digits = 3)))
print(paste0("BRCA.mean.pca is: ", round(BRCA.mean.pca, digits = 3)))
print(paste0("BRCA.mean.raw is: ", round(BRCA.mean.raw, digits = 3)))

#For COAD
colnames(auroc.COAD.data)
COAD.mean.vae <- Logit_mean(auroc.COAD.data$`VAE(650)`)
COAD.mean.pca <- Logit_mean(auroc.COAD.data$`PCA(387)`)
COAD.mean.raw <- Logit_mean(auroc.COAD.data$`Raw(12,097)`)

print(paste0("COAD.mean.vae is: ", round(COAD.mean.vae, digits = 3)))
print(paste0("COAD.mean.pca is: ", round(COAD.mean.pca, digits = 3)))
print(paste0("COAD.mean.raw is: ", round(COAD.mean.raw, digits = 3)))

#For PAAD
colnames(auroc.PAAD.data)
PAAD.mean.vae <- Logit_mean(auroc.PAAD.data$`VAE(50)`)
PAAD.mean.pca <- Logit_mean(auroc.PAAD.data$`PCA(387)`)
PAAD.mean.raw <- Logit_mean(auroc.PAAD.data$`Raw(12,097)`)

print(paste0("PAAD.mean.vae is: ", round(PAAD.mean.vae, digits = 3)))
print(paste0("PAAD.mean.pca is: ", round(PAAD.mean.pca, digits = 3)))
print(paste0("PAAD.mean.raw is: ", round(PAAD.mean.raw, digits = 3)))

#For SARC
colnames(auroc.SARC.data)
SARC.mean.vae <- Logit_mean(auroc.SARC.data$`VAE(500)`)
SARC.mean.pca <- Logit_mean(auroc.SARC.data$`PCA(387)`)
SARC.mean.raw <- Logit_mean(auroc.SARC.data$`Raw(12,097)`[-which(auroc.SARC.data$`Raw(12,097)` %in% NA)])

print(paste0("SARC.mean.vae is: ", round(SARC.mean.vae, digits = 3)))
print(paste0("SARC.mean.pca is: ", round(SARC.mean.pca, digits = 3)))
print(paste0("SARC.mean.raw is: ", round(SARC.mean.raw, digits = 3)))


#calculate the 95% confidence interval with 1000 replications

#For BLCA
colnames(auroc.BLCA.data)
BLCA.lower95.vae <- quantile(replicate(1000, { Logit_mean(sample(auroc.BLCA.data$`VAE(50)`, replace=TRUE)) }), probs=0.025)
BLCA.upper95.vae <- quantile(replicate(1000, { Logit_mean(sample(auroc.BLCA.data$`VAE(50)`, replace=TRUE)) }), probs=0.975)

BLCA.lower95.pca <- quantile(replicate(1000, 
                            { Logit_mean(sample(auroc.BLCA.data$`PCA(387)`[-which(auroc.BLCA.data$`PCA(387)` %in% NA)], replace=TRUE)) }), 
                            probs=0.025)

BLCA.upper95.pca <- quantile(replicate(1000, 
                            { Logit_mean(sample(auroc.BLCA.data$`PCA(387)`[-which(auroc.BLCA.data$`PCA(387)` %in% NA)], replace=TRUE)) }), 
                            probs=0.975)

BLCA.lower95.raw <- quantile(replicate(1000, { Logit_mean(sample(auroc.BLCA.data$`Raw(12,097)`, replace=TRUE)) }), probs=0.025)
BLCA.upper95.raw <- quantile(replicate(1000, { Logit_mean(sample(auroc.BLCA.data$`Raw(12,097)`, replace=TRUE)) }), probs=0.975)

#For BRCA
colnames(auroc.BRCA.data)
BRCA.lower95.vae <- quantile(replicate(1000, { Logit_mean(sample(auroc.BRCA.data$`VAE(50)`, replace=TRUE)) }), probs=0.025)
BRCA.upper95.vae <- quantile(replicate(1000, { Logit_mean(sample(auroc.BRCA.data$`VAE(50)`, replace=TRUE)) }), probs=0.975)
BRCA.lower95.pca <- quantile(replicate(1000, { Logit_mean(sample(auroc.BRCA.data$`PCA(387)`, replace=TRUE)) }), probs=0.025)
BRCA.upper95.pca <- quantile(replicate(1000, { Logit_mean(sample(auroc.BRCA.data$`PCA(387)`, replace=TRUE)) }), probs=0.975)
BRCA.lower95.raw <- quantile(replicate(1000, { Logit_mean(sample(auroc.BRCA.data$`Raw(12,097)`, replace=TRUE)) }), probs=0.025)
BRCA.upper95.raw <- quantile(replicate(1000, { Logit_mean(sample(auroc.BRCA.data$`Raw(12,097)`, replace=TRUE)) }), probs=0.975)

#For COAD
colnames(auroc.COAD.data)
COAD.lower95.vae <- quantile(replicate(1000, { Logit_mean(sample(auroc.COAD.data$`VAE(650)`, replace=TRUE)) }), probs=0.025)
COAD.upper95.vae <- quantile(replicate(1000, { Logit_mean(sample(auroc.COAD.data$`VAE(650)`, replace=TRUE)) }), probs=0.975)
COAD.lower95.pca <- quantile(replicate(1000, { Logit_mean(sample(auroc.COAD.data$`PCA(387)`, replace=TRUE)) }), probs=0.025)
COAD.upper95.pca <- quantile(replicate(1000, { Logit_mean(sample(auroc.COAD.data$`PCA(387)`, replace=TRUE)) }), probs=0.975)
COAD.lower95.raw <- quantile(replicate(1000, { Logit_mean(sample(auroc.COAD.data$`Raw(12,097)`, replace=TRUE)) }), probs=0.025)
COAD.upper95.raw <- quantile(replicate(1000, { Logit_mean(sample(auroc.COAD.data$`Raw(12,097)`, replace=TRUE)) }), probs=0.975)

#For PAAD
colnames(auroc.PAAD.data)
PAAD.lower95.vae <- quantile(replicate(1000, { Logit_mean(sample(auroc.PAAD.data$`VAE(50)`, replace=TRUE)) }), probs=0.025)
PAAD.upper95.vae <- quantile(replicate(1000, { Logit_mean(sample(auroc.PAAD.data$`VAE(50)`, replace=TRUE)) }), probs=0.975)
PAAD.lower95.pca <- quantile(replicate(1000, { Logit_mean(sample(auroc.PAAD.data$`PCA(387)`, replace=TRUE)) }), probs=0.025)
PAAD.upper95.pca <- quantile(replicate(1000, { Logit_mean(sample(auroc.PAAD.data$`PCA(387)`, replace=TRUE)) }), probs=0.975)
PAAD.lower95.raw <- quantile(replicate(1000, { Logit_mean(sample(auroc.PAAD.data$`Raw(12,097)`, replace=TRUE)) }), probs=0.025)
PAAD.upper95.raw <- quantile(replicate(1000, { Logit_mean(sample(auroc.PAAD.data$`Raw(12,097)`, replace=TRUE)) }), probs=0.975)

#For SARC
colnames(auroc.SARC.data)
SARC.lower95.vae <- quantile(replicate(1000, { Logit_mean(sample(auroc.SARC.data$`VAE(500)`, replace=TRUE)) }), probs=0.025)
SARC.upper95.vae <- quantile(replicate(1000, { Logit_mean(sample(auroc.SARC.data$`VAE(500)`, replace=TRUE)) }), probs=0.975)
SARC.lower95.pca <- quantile(replicate(1000, { Logit_mean(sample(auroc.SARC.data$`PCA(387)`, replace=TRUE)) }), probs=0.025)
SARC.upper95.pca <- quantile(replicate(1000, { Logit_mean(sample(auroc.SARC.data$`PCA(387)`, replace=TRUE)) }), probs=0.975)
SARC.lower95.raw <- quantile(replicate(1000, 
                            { Logit_mean(sample(auroc.SARC.data$`Raw(12,097)`[-which(auroc.SARC.data$`Raw(12,097)` %in% NA)], replace=TRUE)) }), 
                            probs=0.025)
SARC.upper95.raw <- quantile(replicate(1000, 
                            { Logit_mean(sample(auroc.SARC.data$`Raw(12,097)`[-which(auroc.SARC.data$`Raw(12,097)` %in% NA)], replace=TRUE)) }), 
                            probs=0.975)


#built a dataframe to store the mean, 95 upper, and 95 lower
mean_95_interval.df <- data.frame(matrix(ncol = 5, nrow = 15))
colnames(mean_95_interval.df) <- c("cancer_type", "ind", "mean", "lower95", "upper95")
mean_95_interval.df$cancer_type <- c("BLCA","BLCA","BLCA", "BRCA", "BRCA", "BRCA", "COAD", "COAD", "COAD",
                             "PAAD", "PAAD", "PAAD", "SARC", "SARC", "SARC")
mean_95_interval.df$ind <- c("Raw(12,097)","PCA(387)","VAE(50)", 
                             "Raw(12,097)","PCA(387)","VAE(50)",
                             "Raw(12,097)","PCA(387)","VAE(650)",
                             "Raw(12,097)","PCA(387)","VAE(50)",
                             "Raw(12,097)","PCA(387)","VAE(500)")
mean_95_interval.df$mean <- c(BLCA.mean.raw, BLCA.mean.pca, BLCA.mean.vae, BRCA.mean.raw, BRCA.mean.pca, BRCA.mean.vae,
                              COAD.mean.raw, COAD.mean.pca, COAD.mean.vae, PAAD.mean.raw, PAAD.mean.pca, PAAD.mean.vae,
                              SARC.mean.raw, SARC.mean.pca, SARC.mean.vae)

mean_95_interval.df$lower95 <- c(BLCA.lower95.raw, BLCA.lower95.pca, BLCA.lower95.vae, BRCA.lower95.raw, BRCA.lower95.pca, BRCA.lower95.vae,
                                COAD.lower95.raw, COAD.lower95.pca, COAD.lower95.vae, PAAD.lower95.raw, PAAD.lower95.pca, PAAD.lower95.vae,
                                SARC.lower95.raw, SARC.lower95.pca, SARC.lower95.vae)
  
  
mean_95_interval.df$upper95 <- c(BLCA.upper95.raw, BLCA.upper95.pca, BLCA.upper95.vae, BRCA.upper95.raw, BRCA.upper95.pca, BRCA.upper95.vae,
                                COAD.upper95.raw, COAD.upper95.pca, COAD.upper95.vae, PAAD.upper95.raw, PAAD.upper95.pca, PAAD.upper95.vae,
                                SARC.upper95.raw, SARC.upper95.pca, SARC.upper95.vae)


################################################
#errorbar plot with confidence interval
###################################################
#reshape data
merged.bind.stack <- merge(auroc.bind.stack, mean_95_interval.df)


#generate the graph
ggplot(merged.bind.stack, aes(x = ind, y = values, color = ind)) + 
  #geom_boxplot(outlier.shape = NA) +
  geom_jitter(shape=16, position=position_jitter(0.2)) +
  labs(y = "AUROC") +
  theme(axis.text = element_text(size = 24), axis.title=element_text(size=36,face="bold"), strip.text.x = element_text(size = 36,face="bold"),
        text = element_text(size =30), panel.background = element_rect(fill = "white", colour = "grey50"), legend.position = "none",
        axis.title.x = element_blank()) +
  scale_y_continuous(breaks = seq(floor(min(auroc.bind.stack$values)), ceiling(max(auroc.bind.stack$values)), by = 0.02) ) +
  geom_errorbar(aes(ymin=lower95, ymax=upper95, color = ind), width=0.5) +
  geom_point(aes(x = ind, y = mean, color = ind), size = 4, shape = 15) +
  facet_wrap(~ cancer_type, scale="free_x", ncol = 2) +
  ggsave("(95ConfidenceInterval)fig_top20_vae_ref_replications(5_cancers, preprint).pdf", width=15, height=20,limitsize = FALSE)
    


####################################
#clean
#######################
#rm(list=ls(all=TRUE))
