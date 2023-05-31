# An R implementation of the Blum Blum Shub Generator by  Lenore Blum, Manuel Blum and Michael Shub(https://www.iacr.org/cryptodb/data/paper.php?pubkey=1751)
# It takes two primes P and Q that are co-primes and a seed S to generate the initial value.
# It is a one way generation function.
blumBlumShubGenerator <- function(s,p,q){
  return((s^2)%%(p*q))
}
library(primes)
library(random)

#Setting the dataPath for storage
dataPath <- "C:/Users/yomaa/OneDrive/University of Chicago/Assignment/Statistical Analytics/RCode/Week3"

#Generating Random Numbers using Random.Org package
trueSample <- randomNumbers(n=2000,min=0,max=100000,col=1)
trueSample <- trueSample[!duplicated(trueSample),]
while(length(trueSample)<1001){
    trueSample.second <- randomNumbers(n=2000,min=0,max=100000,col=1)
    trueSample.second <- trueSample.second[!duplicated(trueSample.second),]
    trueSample <- append(trueSample,trueSample.second)
    trueSample <- trueSample[!duplicated(trueSample),]
}

trueSample <- trueSample[2:1001]
#making the sample uniform, by scaling them down proportionately to the max value.
maxs <- max(trueSample)
trueSample <- ((trueSample*100)/maxs)/100
trueSample <- c(trueSample)

p <- 71413
q <- 83231
seed <- as.numeric(Sys.time())
blumblumshub <- c(seed)
counter <- 0
#Generating the pseudo random numbers.
while(length(blumblumshub)<1001){
  counter <- counter+1
  blumblumshub <- append(blumblumshub,blumBlumShubGenerator(blumblumshub[counter],p,q))
}
blumblumshub <- blumblumshub[2:1001]
maxs <- max(blumblumshub)
blumblumshub <- ((blumblumshub*100)/maxs)/100

res<-data.frame(pseudoSample=blumblumshub,trueSample=trueSample)
saveRDS(res,paste(dataPath,'result.rds',sep='/'))
hist(blumblumshub)
