dataPath<-"C:/Users/yomaa/OneDrive/University of Chicago/Assignment/Statistical Analytics/RCode/Week5"
dat <- read.table(paste(dataPath,'ResidualAnalysisProjectData_2.csv',sep = '/'), header=TRUE,sep = ",")
plot(dat$Input,(dat$Output-mean(dat$Output))^2, type="p",pch=19,
     ylab="Squared Deviations")
nSample<-length(dat$Input)
m1<-lm(Output~Input,dat)
m1$coefficients
yi <- m1$coefficients[1] + m1$coefficients[2]*dat$Input
clusteringParabola <- (yi - mean(dat$Input))^2
points(dat$Input,clusteringParabola,pch=19,col="red")

Unscrambling.Sequence.Steeper.var <- ((dat$Output-mean(dat$Output))^2>clusteringParabola)
head(Unscrambling.Sequence.Steeper.var,10)

Subsample.Steeper.var <- data.frame(steeperInput.var=dat$Input,steeperOutput.var=rep(NA,nSample))
Subsample.Flatter.var <- data.frame(flatterInput.var=dat$Input,flatterOutput.var=rep(NA,nSample))

Subsample.Steeper.var[Unscrambling.Sequence.Steeper.var,2] <- dat[Unscrambling.Sequence.Steeper.var,2]
Subsample.Flatter.var[!Unscrambling.Sequence.Steeper.var,2] <- dat[!Unscrambling.Sequence.Steeper.var,2]

head(cbind(dat,Subsample.Steeper.var,Subsample.Flatter.var),10)

plot(dat$Input,(dat$Output-mean(dat$Output))^2,type="p",pch=19,ylab="Squared Deviations")
points(dat$Input,clusteringParabola,pch=19,col="red")
points(dat$Input[Unscrambling.Sequence.Steeper.var], (dat$Output[Unscrambling.Sequence.Steeper.var]- mean(dat$Output))^2, pch=19,col="blue")
points(dat$Input[!Unscrambling.Sequence.Steeper.var], (dat$Output[!Unscrambling.Sequence.Steeper.var]- mean(dat$Output))^2, pch=19,col="green")
mean(dat$Input)
excludeMiddle<-(dat$Input<mean(dat$Input)-0.5)|(dat$Input>mean(dat$Input)+0.5)
matplot(dat$Input[excludeMiddle],cbind(dat$Output[excludeMiddle],Subsample.Steeper.var$steeperOutput.var[excludeMiddle], Subsample.Flatter.var$flatterOutput.var[excludeMiddle]), type="p",col=c("black","green","blue"), pch=16,ylab="Separated Subsamples")


dat.Steep.var <- lm(steeperOutput.var ~ steeperInput.var,Subsample.Steeper.var[excludeMiddle,])
dat.Flat.var <- lm(flatterOutput.var ~ flatterInput.var,Subsample.Flatter.var[excludeMiddle,])

rbind(Steeper.Coefficients.var=dat.Steep.var$coefficients,
      Flatter.Coefficients.var=dat.Flat.var$coefficients)


plot(dat$Input,dat$Output)
lines(dat$Input,predict(dat.Steep.var,data.frame(steeperInput.var=dat$Input),interval="prediction")[,1],col="red",lwd=3)
lines(dat$Input,predict(dat.Flat.var,data.frame(flatterInput.var=dat$Input),interval="prediction")[,1],col="green",lwd=3)

summary(dat.Steep.var)
estimatedResiduals<-m1$residuals
matplot(dat$Input[excludeMiddle],
        cbind(c(summary(dat.Steep.var)$residuals,
                summary(dat.Flat.var)$residuals),
              estimatedResiduals[excludeMiddle]),
        type="p",pch=c(19,16),ylab="Residuals before and after unscrabling")

res <- list( GeneralModel = m1,mSteep = dat.Steep.var,mFlat = dat.Flat.var)
saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))