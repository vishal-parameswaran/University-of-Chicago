library(benford.analysis)
options(scipen = 999)

dataPath <- "C:/Users/yomaa/OneDrive/University of Chicago/Assignment/Statistical Analytics/RCode/Week6/Group"
test_data <- read.table(paste(dataPath,'test_sample.csv',sep = '/'), header=TRUE)

test_data$x #sample of length 5000

bfd.cp <- benford(test_data$x)
bfd.cp

names(bfd.cp)
head(suspectsTable(bfd.cp),10)

suspectsTable(bfd.cp)[90]

duplicatesTable(bfd.cp)
#(digits_50_and_99 <- getDigits(bfd.cp, test_data$x, digits=c(50, 99)))

bfd.cp$info
bfd.cp$mantissa
#getSuspects(bfd.cp,test_data$x)

Chi_square_p.value <- bfd.cp$stats$chisq$p.value
Mantissa_p.value <- bfd.cp$stats$mantissa.arc.test$p.value

dig<-extract.digits(test_data$x,number.of.digits = 2)
head(dig)

observed<-table(dig[,2])
probObserved<-observed/sum(observed)
head(cbind(observed,probObserved))

dbenford<-sapply(10:99,function(z) p.these.digits(z))
head(dbenford)

chisq.test(x=observed,p=dbenford)

bfd.cp$stats$chisq

MAD <- MAD(bfd.cp)

mean(abs(probObserved-dbenford))

dfactor(bfd.cp)
MAD.conformity <- bfd.cp$MAD.conformity

df <- suspectsTable(bfd.cp, by="absolute.diff")

v1 <- cbind(suspectsTable(bfd.cp, by="difference")[c(89 , 2 , 88 , 5 , 90 , 1),])
#v1 <- v1[order(rank(v1$absolute.diff), decreasing = TRUE), ]
numbers <- array(v1$digits, dim=length(v1$digits))


res <- list(numbers=numbers,
            Chi_square_p.value = Chi_square_p.value,
            Mantissa_p.value = Mantissa_p.value,
            MAD = MAD,
            MAD.conformity  = MAD.conformity
)
saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))

names(bfd.cp$bfd)