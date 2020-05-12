prepare_X_Phi <- function(X_train_orig, X_test_orig, D = 5, spline_type = c("n", "b")){
  
  n_train <- nrow(X_train_orig)
  n_test <- nrow(X_test_orig)
  p <- ncol(X_train_orig)
  
  
  X_train <- matrix(NA, nrow = n_train, ncol = p) # holds centered and scaled covariates
  X_test <- matrix(NA, nrow = n_test, ncol = p)
  Phi_train <- array(NA, dim = c(n_train, D-1, p)) # we end up dropping one term after normalization
  Phi_test <- array(NA, dim = c(n_test, D-1, p))
  

  
  for(j in 1:p){
    if(spline_type == "n") splineTemp_train <- splines::ns(X_train_orig[,j], df = D)
    else if(spline_type == "b") splineTemp_train <- splines::bs(X_train_orig[,j], df = D)
    
    
    
    splineTemp_test  <- predict(splineTemp_train, X_test_orig[,j])
      
    splineTemp_train <- cbind(rep(1, times = n_train), X_train_orig[,j], splineTemp_train)
    splineTemp_test <- cbind(rep(1, times = n_test), X_test_orig[,j], splineTemp_test)
      
    tmp_Phi_train <- matrix(NA, nrow = n_train, ncol = D+1)
    tmp_Phi_test <- matrix(NA, nrow = n_test, ncol = D+1)
      
    tmp_Phi_train[,1] <- splineTemp_train[,1]
    tmp_Phi_test[,1] <- splineTemp_test[,1]
      
    for(d in 2:(D+1)){
      tmpY_train <- splineTemp_train[,d]
      tmpY_test <- splineTemp_test[,d]
      tmpX_train <- tmp_Phi_train[,1:(d-1)]
      tmpX_test <- tmp_Phi_test[,1:(d-1)]
        
      tmp_data_train <- data.frame("Y" = tmpY_train, tmpX_train)
      colnames(tmp_data_train) <- c("Y", paste0("X",1:(d-1)))
      tmp_data_test <- data.frame(tmpX_test)
      colnames(tmp_data_test) <- paste0("X",1:(d-1))
      
      # regress d-th raw basis element onto first d-1 orthognal basis elements
      # since 1st element of orthogonal basis is 1, leave out intercept
      modX <- lm(Y ~ -1 + ., data = tmp_data_train) 
      tmp_Phi_train[,d] <- tmpY_train - predict(modX,newdata = tmp_data_train)
      tmp_Phi_test[,d] <- tmpY_test - predict(modX, newdata = tmp_data_test)
    }
      
    # Now we really don't need the first column of all 1's
    tmp_Phi_train <- tmp_Phi_train[,-1]
    tmp_Phi_test <- tmp_Phi_test[,-1]
    col_norm_train <- apply(tmp_Phi_train, MARGIN = 2, FUN = function(x){sqrt(sum(x*x))})
      
    for(d in 1:ncol(tmp_Phi_train)){
      tmp_Phi_train[,d] <- sqrt(n_train) * tmp_Phi_train[,d]/col_norm_train[d]
      tmp_Phi_test[,d] <- sqrt(n_train) * tmp_Phi_test[,d]/col_norm_train[d]
    }
    
    X_train[,j] <- tmp_Phi_train[,1]
    X_test[,j] <- tmp_Phi_test[,1]
    
    Phi_train[,,j] <-tmp_Phi_train[,2:D]
    Phi_test[,,j] <- tmp_Phi_test[,2:D]
  }
  return(list(X_train = X_train, X_test = X_test, Phi_train = Phi_train, Phi_test = Phi_test))
}



orthogonalize_Phi <- function(Phi){
  n <- dim(Phi)[1]
  D <- dim(Phi)[2]
  p <- dim(Phi)[3]
  
  Phi_tilde <- array(dim = c(n,D,p))
  Qmat <- list()
  Dvec <- list()
  for(j in 1:p){
    SVD <- svd(1/n * crossprod(Phi[,,j]))
    Qmat[[j]] <- SVD$u
    Dvec[[j]] <- SVD$d
    Phi_tilde[,,j] <- Phi[,,j] %*% Qmat[[j]] %*% diag(1/sqrt(Dvec[[j]]))
  }
  
  return(list(Phi_tilde = Phi_tilde, Qmat = Qmat, Dvec = Dvec))
  
}


