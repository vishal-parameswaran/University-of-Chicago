dataPath <- "C:/Users/yomaa/OneDrive/University of Chicago/Assignment/Statistical Analytics/RCode/Week9/Self"
Project.Data <- read.table(paste(dataPath,'Week9_Test_Sample.csv',sep = '/'), header=TRUE)

Project.Data[1:10,]

Data.Levels<-as.numeric(Project.Data[1,])
Project.Data<-Project.Data[-1,]
head(Project.Data)

matplot(Project.Data,type="l")

Project.Data.PCA <- princomp(Project.Data)
names(Project.Data.PCA)

Project.Data.PCA$sdev
Project.Data.PCA$loadings
Project.Data.PCA$center
Project.Data.PCA$scale
Project.Data.PCA$n.obs
Project.Data.PCA$scores
Project.Data.PCA$call



eigen(Project.Data)

Project.Data.PCA1 <- prcomp(Project.Data)
Project.Data.PCA1$sdev
Project.Data.PCA1$rotation
Project.Data.PCA1$center


eig <- (Project.Data.PCA1$sdev)^2

#Get variances of principal components
variance <- eig*100/sum(eig)
variance

#cumulative variances 
cumvar <- cumsum(variance)
cumvar

eig.var.cum <- data.frame(eig=eig, variance=variance, cumvariance=cumvar)
eig.var.cum

summary(Project.Data.PCA)

summary(prcomp(Project.Data[,-1]))

summary(princomp(Project.Data[,-1]))
summary(Project.Data)
summary(lm(Resp~., data=Project.Data))



options(scipen = 999)

#First two questions
summary(prcomp(Project.Data[,-1]))
r_sq <- summary(lm(Resp~., data=Project.Data))
r_sq$r.squared
0.9*r_sq$r.squared

#Last question
Project.Data.PCA1 <- prcomp(Project.Data[,-1])
new_lm <- cbind(Project.Data[1], Project.Data.PCA1$x[,c("PC10")])
new_lm
r_sq_new <- summary(lm(Resp~., data=new_lm))
r_sq_new$r.squared

