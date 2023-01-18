dataPath <- "C://Users//yomaa//Downloads//Assignment//Statistical Analytics"
data <- read.table(paste(dataPath,'Week1_Test_Sample.csv',sep = '/'), header=TRUE)
vList <- sort(unique(data$v))
uList <- unique(data$u)
joint_distribution_creation <- as.data.frame(table(data$u,data$v))
joint_distribution <- matrix(, nrow = length(uList), ncol= length(vList))
for (u in uList){
  for(v in vList ){
    val <- subset(joint_distribution_creation, Var1==u&Var2==v)
    joint_distribution[u,v] = as.integer(val["Freq"])
  }
}
rownames(joint_distribution) <- paste("u",seq(1:length(uList)),sep="")
colnames(joint_distribution) <- paste("v",seq(1:length(vList)),sep="")
joint_distribution <- joint_distribution/length(data$u)
u_Marginal = rowSums(joint_distribution)
v_Marginal = colSums(joint_distribution)
u_Conditional_v = joint_distribution[,4]/v_Marginal[4]
unlist(u_Conditional_v)
v_Conditional_u = joint_distribution[3,]/u_Marginal[3]
unlist(u_Conditional_v)
res <-list(Joint_distribution=joint_distribution,
           u_Marginal = u_Marginal,
           v_Marginal = v_Marginal,
           u_Conditional_v = u_Conditional_v,
           v_Conditional_u = v_Conditional_u          )
saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))