dataPath <- "C:\\Users\\yomaa\\OneDrive\\University of Chicago\\Assignment\\Statistical Analytics"
test_data <- read.table(paste(dataPath,'Simpson_Class.csv',sep = '\\'), header=TRUE)
length_data <- length(test_data$A)
Pab <- length(subset(test_data,A>B)$A)/length_data
Pac <- length(subset(test_data,A>C)$A)/length_data
Pbc <- length(subset(test_data,B>C)$B)/length_data
Pbac5 <- round(length(subset(test_data,B>A&B>C&B==5)$B)/length(subset(test_data,B==5)$B),3)
Pcab4 <- round(length(subset(test_data,C>A&B<C&C==4)$C)/length(subset(test_data,C==4)$B),3)
Pbac <- length(subset(test_data,B>A&B>C)$B)/length_data
Pcab <- length(subset(test_data,C>A&B<C)$C)/length_data
Pabc <- length(subset(test_data,A>B&A>C)$A)/length_data

print(paste("P(B>A,B>C|B=5) = ",Pbac5))
print(paste("P(C>A,C>B|C=4) = ",Pcab4))
print(paste("P(A>B,A>C) = ",Pabc))
print(paste("P(B>A,B>C) = ",Pbac))
print(paste("P(C>A,C>B) = ",Pcab))
print(paste("Which is the most effective drug among A and B before drug C is available on the market? = ",max(c(Pab,1-Pab))))
print(paste("Which is the most effective drug among A and B and C? = ",max(c(Pbac,Pcab,Pabc))))