dataPath<-"C:/Users/yomaa/OneDrive/University of Chicago/Assignment/Statistical Analytics/RCode/Week5"
dat <- read.table(paste(dataPath,'Week5_Test_Sample3.csv',sep = '/'), header=TRUE,sep = " ")
plot(dat$Input,dat$Output, type="p",pch=19)
nSample<-length(dat$Input)
m1<-lm(Output~Input,dat)
m1$coefficients
matplot(dat$Input,cbind(dat$Output,m1$fitted.values),type="p",pch=16,ylab="Sample and Fitted Values")

summary(m1)

estimatedResiduals<-m1$residuals
plot(dat$Input,estimatedResiduals)

Probability.Density.Residuals<-density(estimatedResiduals)
plot(Probability.Density.Residuals,ylim=c(0,.8))
lines(Probability.Density.Residuals$x, dnorm(Probability.Density.Residuals$x,mean=mean(estimatedResiduals),sd=sd(estimatedResiduals)))

# Create NA vectors
Train.Sample<-data.frame(trainInput=dat$Input,trainOutput=rep(NA,nSample))
Train.Sample.Steeper<-data.frame(trainSteepInput=dat$Input, trainSteepOutput=rep(NA,nSample))
Train.Sample.Flatter<-data.frame(trainFlatInput=dat$Input, trainFlatOutput=rep(NA,nSample))

Train.Sample2<-data.frame(trainInput=dat$Input,trainOutput=rep(NA,nSample))
Train.Sample.Steeper2<-data.frame(trainSteepInput=dat$Input, trainSteepOutput=rep(NA,nSample))
Train.Sample.Flatter2<-data.frame(trainFlatInput=dat$Input, trainFlatOutput=rep(NA,nSample))

# Create selectors
Train.Sample.Selector <- (dat$Input>=2)
Train.Sample.Steeper.Selector<-Train.Sample.Selector& (dat$Output>m1$fitted.values)
Train.Sample.Flatter.Selector<-Train.Sample.Selector& (dat$Output<=m1$fitted.values)

Train.Sample[Train.Sample.Selector,2]<-dat[Train.Sample.Selector,1]
Train.Sample.Steeper[Train.Sample.Steeper.Selector,2]<-dat[Train.Sample.Steeper.Selector,1]
Train.Sample.Flatter[Train.Sample.Flatter.Selector,2]<-dat[Train.Sample.Flatter.Selector,1]

plot(Train.Sample$trainInput,Train.Sample$trainOutput,pch=16,ylab="Training Sample Output", xlab="Training Sample Input")
points(Train.Sample.Steeper$trainSteepInput,Train.Sample.Steeper$trainSteepOutput,pch=20,col="green")
points(Train.Sample.Flatter$trainFlatInput,Train.Sample.Flatter$trainFlatOutput,pch=20,col="blue")

Train.Sample.Steep.lm <- lm(trainSteepOutput~trainSteepInput,Train.Sample.Steeper)
Train.Sample.Flat.lm <- lm(trainFlatOutput~trainFlatInput,Train.Sample.Flatter)
plot(dat$Input,dat$Output, type="p",pch=19)
lines(dat$Input,predict(Train.Sample.Steep.lm,data.frame(trainSteepInput=dat$Input),interval="prediction")[,1],col="red",lwd=3)
lines(dat$Input,predict(Train.Sample.Flat.lm,data.frame(trainFlatInput=dat$Input),interval="prediction")[,1],col="green",lwd=3)


Distances.to.Steeper<-abs(dat$Output-dat$Input*Train.Sample.Steep.lm$coefficients[2]-Train.Sample.Steep.lm$coefficients[1])
Distances.to.Flatter<-abs(dat$Output-dat$Input*Train.Sample.Flat.lm$coefficients[2]-Train.Sample.Flat.lm$coefficients[1])
Unscrambling.Sequence.Steeper<-Distances.to.Steeper<Distances.to.Flatter
Subsample.Steeper<-data.frame(steeperInput=dat$Input,steeperOutput=rep(NA,nSample))
Subsample.Flatter<-data.frame(flatterInput=dat$Input,flatterOutput=rep(NA,nSample))

Subsample.Steeper[Unscrambling.Sequence.Steeper,2]<-dat[Unscrambling.Sequence.Steeper,1]
Subsample.Flatter[!Unscrambling.Sequence.Steeper,2]<-dat[!Unscrambling.Sequence.Steeper,1]

matplot(dat$Input,cbind(dat$Output,
                        Subsample.Steeper$steeperOutput,
                        Subsample.Flatter$flatterOutput),
        type="p",col=c("black","green","blue"),
        pch=16,ylab="Separated Subsamples")
mSteep <- lm(steeperOutput~steeperInput,Subsample.Steeper)
mFlat <- lm(flatterOutput~flatterInput,Subsample.Flatter)
rbind(Steeper.Coefficients=mSteep$coefficients,
      Flatter.Coefficients=mFlat$coefficients)

matplot(dat$Input,cbind(c(summary(mSteep)$residuals,
                          summary(mFlat)$residuals),
                        estimatedResiduals),type="p",pch=c(19,16),ylab="Residuals before and after unscrambling")
legend("bottomleft",legend=c("Before","After"),col=c("red","black"),pch=16)
res <- list( GeneralModel = m1,mSteep = mSteep,mFlat = mFlat)
saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))