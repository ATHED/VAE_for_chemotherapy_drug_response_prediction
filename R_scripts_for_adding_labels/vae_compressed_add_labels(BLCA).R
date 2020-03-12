###########################################
#read in vae compressed data
#################################################
#expression data
expr.data <- read.table("/media/qiwei/work/Python_playground/VAE/TCGA_5_cancers/counts_data/vae_compressed/encoded_5cancers_rnaseq_vae(perSp,a1.0,unlabel,0.2_var,2LF6k,z50,minmax).tsv", sep="\t",
                            header=T, stringsAsFactors=FALSE, row.names = 1,
                            quote="",
                            comment.char="#")

#################################
#clinical data
cli.data <- read.table ("/media/qiwei/work/Python_playground/VAE/TCGA_5_cancers/clinical_data/TCGA-BLCA-binary-labels.txt", sep="\t",
                        header=T, stringsAsFactors=FALSE,
                        quote="",
                        comment.char="#")
#colnames(cli.data) <- cli.data[1,]
cli.data <- cli.data[,-1]

#select only therapy_type and measure of response
colnames(cli.data)


#######################################
#merged dataframe
#expression data dim 1:60484
expr.temp.data <- expr.data #data.frame(t(expr.temp2.data))

expr.temp.data$Ensembl_ID <- rownames(expr.temp.data)
#expr.temp.data[,60484]

merge.data <- merge(expr.temp.data, cli.data, by.x = "Ensembl_ID", by.y = "submitter_id.samples", all.x = T)
colnames(merge.data)
#which(merge.data$therapy_type == "Chemotherapy")

##if using only labeled tumor samples then don not need to do the subsetting
#merge.chemo.full.data <- merge.chemo.data

#double check the # of positive/negative response
nrow(merge.data[which(merge.data$response_group == 1), ])
nrow(merge.data[which(merge.data$response_group == 0), ])



merge.chemo.label.data <- merge.data[-which(is.na(merge.data$response_group)),]

#merge.response.data <- merge.chemo.full.data[,c(1:(ncol(merge.chemo.full.data) - 3), ncol(merge.chemo.full.data))]

colnames(merge.chemo.label.data[,(ncol(merge.chemo.label.data) - 6):ncol(merge.chemo.label.data)])

#########################################################
#output file
#################################################################
write.table(merge.chemo.label.data,
            file="/media/qiwei/work/Python_playground/VAE/TCGA_5_cancers/counts_data/vae_compressed_wLabels/encoded_BLCA_rnaseq_vae_binaryLabels(perSp,a1.0,unlabel,0.2_var,2LF6k,z50,minmax).txt",
            sep="\t",
            quote=FALSE,
            row.names=T,
            col.names=NA)



####################################
#clean
#######################
#rm(list=ls(all=TRUE))
