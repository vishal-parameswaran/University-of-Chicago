dataPath <- "C:/Users/yomaa/OneDrive/University of Chicago/Assignment/Statistical Analytics/"
df <- read.table(paste0(dataPath, 'Week2_Test_Sample.csv'), header=TRUE)
sdX <- round(sd(df$x),2)
sdY <- round(sd(df$y),2)
cXY <- round(cor(df$x,df$y),2)
a <- (sdY/sdX)*cXY
result <- data.frame(sdX=sdX, sdY=sdY, cXY=cXY, a=a)
write.table(result, file = paste(dataPath,'result.csv',sep = '/'), row.names = F)