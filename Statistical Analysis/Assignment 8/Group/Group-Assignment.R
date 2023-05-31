datapath<- "C:/Users/yomaa/OneDrive/University of Chicago/Assignment/Statistical Analytics/RCode/Week8/Group"
AssignmentData<-
  read.csv(file=paste(datapath,
                      "test_sample.csv",sep="/"),
           row.names=1,header=TRUE,sep=",")[,1:8]
head(AssignmentData)

matplot(AssignmentData[,c(1:7)],type='l',ylab="Interest Rates",
        main="History of Interest Rates",xlab="Index")
matplot(AssignmentData[,],type='l',ylab="Interest Rates",
        main="History of Interest Rates and Output",xlab="Index")
Window.width<-20; Window.shift<-5
library(zoo)
all.means<-rollapply(AssignmentData,width=Window.width,
                     by=Window.shift,by.column=TRUE, sd)
head(all.means)
Count<-1:dim(AssignmentData)[1]
Rolling.window.matrix<-rollapply(Count,
                                 width=Window.width,
                                 by=Window.shift,
                                 by.column=FALSE,FUN=function(z) z)
Rolling.window.matrix[1:10,] 

Points.of.calculation<-Rolling.window.matrix[,10]    
Points.of.calculation[1:10]

length(Points.of.calculation)
Means.forPlot<-rep(NA,dim(AssignmentData)[1])
Means.forPlot[Points.of.calculation]<-all.means[,1]
all.means <- as.data.frame(all.means)
all.means["counts"] <-  Points.of.calculation
all.means
for(i in 1:nrow(all.means)){
  if(all.means[i,8]>=0.3){
    if(i==1){
      
    }
  }
}
