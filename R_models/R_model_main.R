R_model_main = function() {

  # INIT --------------------------------------------------------------------
    
    if (!require("rstudioapi")) install.packages("rstudioapi"); library("rstudioapi")
    if (!require("whitening")) install.packages("whitening"); library("whitening")
    
    full_path = rstudioapi::getActiveDocumentContext()$path
    script.dir = strsplit(full_path, split = .Platform$file.sep, fixed=TRUE)[[1]]
    just_the_file = tail(script.dir,1)    
    script.dir = gsub(just_the_file, '', full_path)
      
    # remove the last separator
    if (substr(script.dir, nchar(script.dir), nchar(script.dir)) == '/') {
      script.dir = substr(script.dir, 1, nchar(script.dir)-1)
    } else if (substr(script.dir, nchar(script.dir), nchar(script.dir)) == '/') {
      script.dir = substr(script.dir, 1, nchar(script.dir)-1)
    }
    
  # IMPORT --------------------------------------------------------------------
    
    data_path = file.path(script.dir, '..', 'test_data_private', 'df_scaled.csv')
    data_in = read.csv(data_path, stringsAsFactors = FALSE)
    variable_type = unname(unlist(data_in[1,]))
    
    data_wo_headers = data_in[2:dim(data_in)[1],] # drop the first row
    subheaders = unlist(data_in[1,])
    
    data_continuous = list()
    data_categorical = list()
    col_names = colnames(data_wo_headers)
    
    cont_labels = list()
    categ_labels = list()
    
    # data is now string and not double
    indices_cont = variable_type %in% 'continuous'
    for (i in 1 : dim(data_wo_headers)[2]) {
      
      vector_in = unlist(data_wo_headers[,i])
      
      if (identical(variable_type[i], 'continuous')) {
        
        data_wo_headers[,i] = as.double(data_wo_headers[,i])
        data_continuous[[col_names[i]]] = vector_in
        cont_labels = c(cont_labels, col_names[i])
        
      } else if (identical(variable_type[i], 'categorical')) {
        
        data_wo_headers[,i] = as.integer(data_wo_headers[,i])
        data_categorical[[col_names[i]]] = vector_in
        categ_labels = c(categ_labels, col_names[i])
      }
    }
    
    # TODO! use the key
    data = data_wo_headers[, 1:(dim(data_in)[2]-2)]
    regression_label = as.double(unlist(data_wo_headers[, dim(data_in)[2]]))
    classif_label = as.integer(unlist(data_wo_headers[, dim(data_in)[2]-1]))
    
    # convert to matrix 
    no_of_cols = length(data_continuous)
    no_of_samples = length(data_continuous[[1]])
    data_continuous_mat = matrix(, nrow = no_of_samples, ncol = no_of_cols)
    for (c in 1 : no_of_cols) {
      data_continuous_mat[,c] = as.double(unlist(data_continuous[[c]]))
    }
    
    # convert to matrix 
    no_of_cols = length(data_categorical)
    data_categorical_mat = matrix(, nrow = no_of_samples, ncol = no_of_cols)
    for (c in 1 : no_of_cols) {
      data_categorical_mat[,c] = as.integer(unlist(data_categorical[[c]]))
    }
    
    
  # Transform somehow
    
      # As a result we recommend two particular approaches: ZCA-cor whitening to produce sphered variables 
      # that are maximally similar to the original variables, and PCA-cor whitening to obtain sphered variables 
      # that maximally compress the original variables.
      # https://doi.org/10.1080/00031305.2016.1277159
      data_transformed = transform.data(data_in = data_continuous_mat,
                                        method="ZCA-cor")
      
  # Write to disk with the untransformed categorical variables
      
      output_dir = file.path(script.dir, '..', 'test_data_private')
      
      write_to_disk(continuous_mat = data_continuous_mat,
                    categorical_mat = data_categorical_mat,
                    classif_label = classif_label,
                    regression_label = regression_label,
                    cont_labels = cont_labels,
                    categ_labels = categ_labels,
                    headers = col_names,
                    subheaders = subheaders,
                    output_dir = output_dir,
                    tr_method = 'z')
      
      write_to_disk(continuous_mat = data_transformed,
                    categorical_mat = data_categorical_mat,
                    classif_label = classif_label,
                    regression_label = regression_label,
                    cont_labels = cont_labels,
                    categ_labels = categ_labels,
                    headers = col_names,
                    subheaders = subheaders,
                    output_dir = output_dir,
                    tr_method = 'ZCA')
}
    
transform.data = function(data_in, method) {
  
  Z.ZCAcor.2 = whiten(data_in, method=method)
  zapsmall( cov(Z.ZCAcor.2) )
  return(Z.ZCAcor.2)
  
}

write_to_disk = function(continuous_mat,
                          categorical_mat,
                          classif_label,
                          regression_label,
                          cont_labels,
                          categ_labels,
                          headers,
                          subheaders,
                          output_dir,
                          tr_method) {
  
  # combine the data
  data_comb = cbind(continuous_mat, categorical_mat, classif_label, regression_label)
  data_cont = cbind(continuous_mat, classif_label, regression_label)
  colnames(data_comb) = headers
  cont_labels = unlist(cont_labels)
  cont_labels_2 = c(cont_labels, 'classification label', 'regression label')
  colnames(data_cont) = cont_labels_2
  
  df = data.frame(data_comb)
  file = paste0('df_', tr_method, '_cor.csv')
  write.csv(df, file = file.path(output_dir, file), row.names = FALSE)
  
  df_cont = data.frame(data_cont)
  file = paste0('df_cont_', tr_method, '_cor.csv')
  write.csv(df_cont, file = file.path(output_dir, file), row.names = FALSE)
  
  
}
      
    
    
 