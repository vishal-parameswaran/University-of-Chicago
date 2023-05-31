dataPath<-"C:/Users/yomaa/OneDrive/University of Chicago/Assignment/Statistical Analytics/RCode/Week4"
dat <- read.table(paste(dataPath,'Week4_Test_Sample.csv',sep = '/'), header=TRUE)

Estimated_LinearModel <- lm(Y ~ X,data=dat)
Estimated_Residuals <- Estimated_LinearModel$residuals

Unscrambled.Selection.Sequence <- as.integer(as.logical(Estimated_Residuals>0))
res <- list(Unscrambled.Selection.Sequence =  Unscrambled.Selection.Sequence)
write.table(res, file = paste(dataPath,'result.csv',sep = '/'), row.names = F)