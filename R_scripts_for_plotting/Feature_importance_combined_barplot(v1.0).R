library(ggplot2)
#########################################
#calculate 95% confidence interval
##########################################
#expression data
#rank_features_pca.data <- read.table("/media/qiwei/work/Python_playground/VAE/TCGA_5_cancers/counts_data/pca_compressed/Ranking_PC_importance(5cancer_COAD).txt"
#                                       , sep=",",
#                                       header=T, stringsAsFactors=FALSE, row.names = 1,
#                                       quote="",
#                                       comment.char="#")

#pca ranking features
rank_features_pca.data <- read.table("Ranking_PC_importance(5cancer_COAD_opCover).txt"
                                     , sep="\t",
                                     header=T, stringsAsFactors=FALSE, row.names = 1,
                                     quote="",
                                     comment.char="#")
colnames(rank_features_pca.data)
rank20_features_pca.data <- rank_features_pca.data[1:20,]
rank20_features_pca.data$Rank_of_features <- rownames(rank20_features_pca.data)
#overide to be 1:20
rank20_features_pca.data$Rank_of_features <- c(1:20)

#vae ranking features
rank_features_vae.data <- read.table("Ranking_VAE_feature_importance(5cancer_COAD_opCover).txt"
                                     , sep="\t",
                                     header=T, stringsAsFactors=FALSE, row.names = 1,
                                     quote="",
                                     comment.char="#")
colnames(rank_features_vae.data)
rank20_features_vae.data <- rank_features_vae.data[1:20,]
rank20_features_vae.data$Rank_of_features <- rownames(rank20_features_vae.data)
#overide to be 1:20
rank20_features_vae.data$Rank_of_features <- c(1:20)

###############################################
#change column names and combined those data
#############################################
rank20.pca <- rank20_features_pca.data[,c(3,2)]
rank20.pca$Group <- "PCA"
colnames(rank20.pca) <- c("Rank of features","Sum of importance", "Group")

rank20.vae <- rank20_features_vae.data[,c(3,2)]
rank20.vae$Group <- "VAE"
colnames(rank20.vae) <- c("Rank of features","Sum of importance", "Group")

rank20.combined <- rbind(rank20.pca, rank20.vae)
#rank20.combined$`Rank of features` <- factor(rank20.combined$`Rank of features`)

###################################
#Grouped bar plot
###################################


pdf("Top20_extracted_feature_importance(COAD, 5_cancers, pca_vs_vae, opCover).pdf", width=5, height=6)
#par(mar=c(5,6,4,1)+.1)
#barplot(rank20.combined, 
#        horiz = T, ylab = "Rank of feature",  names.arg = rank20.combined$`Rank of features`, 
#        xlim = c(0,700), cex.axis=1.5, cex.names=1.0, cex.main = 1.5, cex.lab = 1.5)
#title(xlab = "Sum of feature importance \nacross 30 replications", line = 4, cex.lab = 1.5)
# Grouped
ggplot(rank20.combined, ylab = "Rank of features", xlab = "Sum of importance",
       xlim = c(0,700), 
       aes(fill=Group, y=`Sum of importance`, x=`Rank of features`)) + 
       scale_fill_grey(start = 0.3, end = 0.7) +
       theme_classic(base_size = 26) +  
       scale_x_discrete(limits=c(1:20)) +
       geom_bar(position="dodge", stat="identity") +
       coord_flip()
dev.off()

####################################
#clean
#######################
#rm(list=ls(all=TRUE))