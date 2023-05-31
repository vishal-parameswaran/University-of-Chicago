dataPath <- "C:/Users/yomaa/OneDrive/University of Chicago/Assignment/Statistical Analytics/"
data3way<-read.csv(file=paste(dataPath,"3WayData.csv",sep="/"))
mat.u0<-table(subset(data3way,u==0)[,1],subset(data3way,u==0)[,2])
mat.u1<-table(subset(data3way,u==1)[,1],subset(data3way,u==1)[,2])
idx.v1<-data3way$v==1
idx.w1<-data3way$w==1
idx.u1<-data3way$u==1
sum(idx.v1*idx.w1*idx.u1)
idx.v2<-data3way$v==2
sum(idx.v2*idx.w1*idx.u1)
colnames(mat.u1)<-colnames(mat.u0)<-c("v1","v2","v3")
rownames(mat.u1)<-rownames(mat.u0)<-c("w1","w2","w3")
data3way.array<-array(rep(NA,18),dim=c(3,3,2),dimnames=list(paste("w",1:3,sep=""),paste("v",1:3,sep=""),paste("u",0:1,sep="")))
data3way.array[,,1]<-mat.u0
data3way.array[,,2]<-mat.u1
N<-sum(data3way.array)
(data3way.array.p<-data3way.array/N)
data3way.array.p
rowSums(data3way.array.p)
vMarginal <- rowSums(colSums(data3way.array.p))
uMarginal <- colSums(colSums(data3way.array.p))
wMarginal <- rowSums(data3way.array.p)
cond.v.w.given.u1 <- data3way.array.p[,,2]/uMarginal[2]
cond.v.given.u1 <- colSums(data3way.array.p[,,2])/uMarginal[2]
cond.w.given.u1.v2 <- data3way.array.p[,2,2]/(cond.v.given.u1[2]*uMarginal[2])
  res<- list(vMarginal = vMarginal,
             uMarginal = uMarginal,
             wMarginal = wMarginal,
              cond1 = cond.v.w.given.u1,
              cond2 = cond.v.given.u1,
              cond3 = cond.w.given.u1.v2)
saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))