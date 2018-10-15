#################### Preparation ##############################
library(mclust)
library(MASS)
library(plyr)
library(dplyr)
library(beepr)
library(dlm)
library(Rcpp)

buildSu <- function(par, Vpar, N){
  
  Ncluster  <- (length(par) - N) / 2
  mu <- par[1:Ncluster]
  std_observation <- par[(Ncluster+1):(Ncluster + N)] 
  std_state <-  par[(Ncluster + N+1):length(par)] 
  
  out <- list(m0 = mu,
              C0 = 1e7 * diag(Ncluster), #fixme increase runtime by giving a better estimate of C0
              FF = Vpar,
              GG = diag(rep(1, Ncluster)),
              V = diag(std_observation^2), # diagonal correlation matrix for observation eqn
              W = diag(std_state^2) # diagonal correlation matrix for state eqn
  )
  return(out)
}


set.seed(1234)
n <- 300
t <-3
counter <- 3
it <- 1

y <- NULL
mu_temp <- NULL
sigma_temp <- NULL
mu_final <- NULL
sigma_final <- NULL
mu_check <- NULL

##############################################################

#################### Simulation ##############################
sigma <- list(
  list(matrix(c(10,0,0,3),2,2),
       matrix(c(3,0,0,10),2,2),
       matrix(c(2,0,0,7),2,2)),
  
  list(matrix(c(9,0,0,3),2,2),
       matrix(c(3,0,0,10),2,2),
       matrix(c(3,0,0,6),2,2)),
  
  list(matrix(c(8,0,0,4),2,2),
       matrix(c(3,0,0,10),2,2),
       matrix(c(5,0,0,5),2,2))
)

mu <- list(
  list(c(-15,0),
       c(-2,2),
       c(10,15)),
  
  list(c(-12,0),
       c(-2,3),
       c(13,13)),
  
  list(c(-10,0),
       c(-3,3),
       c(15,10))
)

data1 <- data.frame()
for(i in 0:2){
  temp1 <- unlist(mu)
  temp2 <- unlist(sigma)
  temp3 <- rbind(mvrnorm(n/3, temp1[(1+6*i):(2+6*i)], matrix(temp2[(1+12*i):(4+12*i)],2)), 
                 mvrnorm(n/3, temp1[(3+6*i):(4+6*i)], matrix(temp2[(5+12*i):(8+12*i)],2)), 
                 mvrnorm(n/3, temp1[(5+6*i):(6+6*i)], matrix(temp2[(9+12*i):(12+12*i)],2)))
  data1 <- rbind(data1, temp3)
  remove(temp1,temp2,temp3)
}

data1$t = as.vector(c(rep(1,n),rep(2,n),rep(3,n))) 
data1$name = as.vector(rep(c(1:(n)), 3))
plot(data1$V1, data1$V2)

##############################################################


################ Step 0 - Initialization ####################

#Now that we have our example data, let's execute a first clustering using Mclust
clust1 <- Mclust(subset(data1, select = c(V1, V2)), G = 3, modelNames = "VVI")
plot(clust1, what = "classification")

data1 <- cbind(data1, clust1$z)

#Make sure the matrix y is in the right format (TxN)
for(i in unique(data1$name)){
  temp <- as.matrix(subset(data1, select = c(V1, V2), name == i))
  if(length(temp)==6){
    y <-  cbind(y, as.vector(temp))
  }
}
y <- rbind(cbind(y[1:t,],matrix(0L, t, n)),cbind(matrix(0L, t, n),y[(t+1):nrow(y),]))

##############################################################


################ Step 2 - Updating parameters ################

while (counter > 0){ #This while-loop runs over step 1 and 2 until a prespecified number of iterations is completed
  
  t1 <- Sys.time() #Save starting time of the iteration to keep track of iteration time
  
  
  #Now for each individual we calculate the average cluster membership likelihoods 
  Vpar <- subset(data1, select = c(`1`,`2`,`3`,`name`)) %>%
    group_by(name)%>%
    summarise_all(funs(mean(.,na.rm = T)))
  
  #And put the highest colum to 1   (I know this code looks overcomplicated, but I've been wrestling with it for a while and 
  #it is the best I could come up with, at least it's fast)
  Vpar$max <- apply(Vpar[,2:4], MARGIN = 1, max) 
  Vpar[,2:4] <- ifelse(Vpar[,2:4]==Vpar$max, 1, 0)
  Vpar = as.matrix(subset(Vpar, select = c(`1`, `2`, `3`)))
  Vpar <- rbind(cbind(Vpar, matrix(0L,n,clust1$G)),cbind(matrix(0L,n,clust1$G), Vpar))
  
  #Straight from the example code:
  N = ncol(y)
  T = nrow(y)
  Ncluster = ncol(Vpar)
  
  #Insert the parameter estimates from the EM clustering as new initial guess
  Mus = matrix(as.vector(t(clust1$parameters$mean)),1)
  StateStdDev = matrix(c(as.vector(clust1$parameters$variance$sigma)[seq(1,4*clust1$G,4)],as.vector(clust1$parameters$variance$sigma)[seq(4,4*clust1$G,4)]),1)
  Hs <- matrix(rep(1, N), 1)
  par_init <- c(Mus, Hs, StateStdDev)
  
  #Use the parameter estimates in MLE to estimate the observation and state variances
  #add/remove capture.output
  capture.output(hoMLE <- dlmMLE(y, par_init, buildSu, Vpar = Vpar, N = N,
                                 debug = TRUE, control = list(trace = TRUE, maxit = 2), method = "SANN"));
  
  #Store the parameter estimates from the MLE in a new variable
  par2 <- hoMLE$par
  #Use the parameter estimates to estimate the cluster means at each point in time using a Kalman Smoother
  hoSmooth <- dlmSmooth(y, dlm(buildSu(par2, Vpar, N)))
  
  #Store the means of the cluster means from the Kalman filter to be used in the next EM iteration
  mu_temp <- c(colMeans(hoSmooth$s[2:(clust1$G+1), 1:clust1$G]), colMeans(hoSmooth$s[(clust1$G+2):nrow(hoSmooth$s), (clust1$G+1):ncol(hoSmooth$s)]))
  
  #Store the results from the MLE estimation of the state variances to be used in the next EM iteration
  sigma_temp <- hoMLE$par[(Ncluster + N+1):length(hoMLE$par)]
  
  mu_check <- rbind(mu_check, round(as.matrix(hoSmooth$s[2:nrow(hoSmooth$s),]), 2)) #(just to store all the time varying results in a matrix to compare after the algorithm is completed)
  
  ##############################################################
  
  ################ Step 1 - Cluster membership ################
  #Insert the updated parameter estimates into the EM algorithm
  clust1$parameters$mean[1,1] <- mean(mu_temp[1])
  clust1$parameters$mean[1,2] <- mean(mu_temp[2])
  clust1$parameters$mean[1,3] <- mean(mu_temp[3])
  clust1$parameters$mean[2,1] <- mean(mu_temp[4])
  clust1$parameters$mean[2,2] <- mean(mu_temp[5])
  clust1$parameters$mean[2,3] <- mean(mu_temp[6])
  
  #fixme variance estimates blow up if squared?!?!?!
  clust1$parameters$variance$sigma[[1]] <- sigma_temp[1]
  clust1$parameters$variance$sigma[[5]] <- sigma_temp[2]
  clust1$parameters$variance$sigma[[9]] <- sigma_temp[3]
  clust1$parameters$variance$sigma[[4]] <- sigma_temp[4]
  clust1$parameters$variance$sigma[[8]] <- sigma_temp[5]
  clust1$parameters$variance$sigma[[12]] <- sigma_temp[6]
  
  #Save the outcome of mu and sigma for each iteration to check after the algorithm has finished
  #Note these are the time-averaged results from the time varying estimates!!
  mu_final <- rbind(mu_final, mu_temp)
  sigma_final <- rbind(sigma_final, sigma_temp)
  
  #Rerun the EM algorithm with the updated parameter estimates
  clust1 <- em(clust1$modelName, subset(data1, select = c(V1,V2)), parameters = clust1$parameters, control = emControl(itmax = 1)) 
  
  #Replace the old likelihoods with the new, to be used to determine the new Vpar matrix in the next iteration
  data1 <- data1[,1:4]
  data1 <- cbind(data1, clust1$z)
  
  #Report iteration is complete and update counter
  cat("\n Iteration ", it,"  is complete, it took: \n", Sys.time()-t1, "\n")
  counter = counter-1
  it = it+1
}
##############################################################
classification <- colnames(data1[,5:7])[apply(data1[,5:7],1,which.max)]
classification

for (i in 1:t){
  plot(data1[data1$t == i,1:2], col = classification, xlim = c(-20, 20), ylim = c(-10,20))
}


plot(clust1)
