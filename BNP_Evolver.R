# BNP Random Evolver
# Made available under Apache 2.0 License
# By Andy White

# This script evolves Generation X-1 into Generation X
# It saves each set of predictions as a CSV
# and the parameters and error scores in a metadata CSV file

# Leave running overnight and then run all features through
# a master model in the morning!

# Load required libraries
library(data.table)
library(Matrix)
library(xgboost)

######################################################################################
## SETUP PARAMETERS (Set these for each session)
######################################################################################
setwd("C:/Users/Andrew.000/Desktop/MSc Data Science and Analytics/KAGGLE/BNP Paribas")
predictions_file_prefix = "base_GEN010_"
metadata_file = "base_metadata_all_gens.csv"
number_of_parents = 100
number_of_children = 100
mutation_probability = 0.125
######################################################################################

# Function for calculating logloss
quicklogloss = function(preds, actual) {
  preds[preds==1] = 0.999999
  preds[preds==0] = 0.000001
  logloss = sum(actual * log(preds) + (1 - actual) * log(1 - preds)) / -length(preds)
  return(logloss)
}

# Load data and mark trainA and trainB splits
train_dt = fread("train.csv", na.strings=c(NA, "", "NA"))
test_dt = fread("test.csv", na.strings=c(NA, "", "NA"))
train_dt[,train_test := "train"]
test_dt[,train_test := "test"]
all_dt = rbind(train_dt, test_dt, fill=T)
rm(list=c("train_dt", "test_dt"))
setkey(all_dt, ID)
n = nrow(all_dt)
trainA = sample(all_dt[train_test=="train",ID], floor(length(all_dt[train_test=="train",ID]) / 2))
all_dt[train_test=="train" & ID %in% trainA, basesplit := "trainA"]
all_dt[train_test=="train" & !(ID %in% trainA), basesplit := "trainB"]
all_dt[train_test=="test", basesplit := "test"]
variable_names = setdiff(names(all_dt), c("ID", "target", "train_test", "basesplit"))
variable_types = rep("num", length(variable_names))
variable_types[sapply(all_dt, is.character)[variable_names]] = "cat"
variable_types[sapply(all_dt, is.integer)[variable_names]] = "int"
names(variable_types) = variable_names

# Load metadata
metadata = fread(metadata_file)
parent_max = sort(metadata[,max_logloss])[number_of_parents]
parent_rows = which(metadata[,max_logloss] <= parent_max)
random_seeds = sample(1:5000, number_of_children)

for (i in 1:number_of_children) {
  set.seed(random_seeds[i])
  print.noquote(paste("Random model:", i))
  # Set up file names
  file_suffix = formatC(i, width=3, flag="0")
  predictions_filename = paste(predictions_file_prefix, file_suffix, ".csv", sep="")
  
  # Choose random parameters
  # These are: -variable choice (always 8 variables)
  #            -treat integers as categoricals?
  #            -include 2-way products of numerics?
  #            -include 2-way sums of numerics?
  #            -include 2-way differences of numerics?
  #            -include 2-way quotients of numerics? (N.B. order matters so this is calculated in both directions)
  #            -rare value cut off [0:1000]
  #            -include probabilities of categoricals?
  #            -include probabilities of binned numerics?
  #            -XGB params:
  #            - max_depth [4:30]
  #            - subsample [0.4:1.0]
  #            - colsample_bytree [0.4:1.0]
  #            - eta [0.01:0.20]
  #            - rounds [50:500]
  #            - seed[0:2000]
  
  # Set up parents
  parents = sample(parent_rows, 2)
  parent_vars = c(as.character(subset(metadata[parents[1],], select=paste("Variable", 1:10))),
                  as.character(subset(metadata[parents[2],], select=paste("Variable", 1:10))))
  parent_vars = parent_vars[parent_vars != "NA" & parent_vars != ""]
  
  parent_inclusions = as.logical(subset(metadata[parents[1],],
                                        select=c("ints_as_cats", "prods", "sums", "diffs", "quots", "probs", "num_probs"))) |
    as.logical(subset(metadata[parents[2],], select=c("ints_as_cats", "prods", "sums", "diffs", "quots", "probs", "num_probs")))
  inclusion_mutations = sample(c(TRUE, FALSE), 7, c(mutation_probability, 1-mutation_probability), replace=T)
  parent_inclusions[inclusion_mutations] = !parent_inclusions[inclusion_mutations] # Random switch of inherited gene
  selected_vars = sample(c(parent_vars, variable_names), 10, prob=c(rep((1-mutation_probability) / length(parent_vars), length(parent_vars)),
                                                                   rep(mutation_probability / 131, 131)))
  while(length(unique(selected_vars)) < 10) {
    selected_vars = unique(selected_vars)
    selected_vars = c(selected_vars, sample(c(parent_vars, variable_names), 1, prob=c(rep((1-mutation_probability) / length(parent_vars), length(parent_vars)),
                                                                                      rep(mutation_probability / 131, 131))))
  }
  selected_types = variable_types[selected_vars]
  inclusions = parent_inclusions
  rare_cutoff = sample(c(metadata[parents[1],rare_cutoff],
                         metadata[parents[2],rare_cutoff],
                         0:1000), 1, prob=c((1-mutation_probability)/2, (1-mutation_probability)/2, rep(mutation_probability / 1001, 1001)))
  r_max_depth = sample(c(metadata[parents[1],xgb_depth],
                         metadata[parents[2],xgb_depth],
                         4:30), 1, prob=c((1-mutation_probability)/2, (1-mutation_probability)/2, rep(mutation_probability / 27, 27)))
  r_subsample = sample(c(metadata[parents[1],xgb_subsample],
                         metadata[parents[2],xgb_subsample],
                         seq(0.4, 1, 0.05)), 1, prob=c((1-mutation_probability)/2, (1-mutation_probability)/2, rep(mutation_probability / 13, 13)))
  r_colsample = sample(c(metadata[parents[1],xgb_colsample],
                         metadata[parents[2],xgb_colsample],
                         seq(0.4, 1, 0.05)), 1, prob=c((1-mutation_probability)/2, (1-mutation_probability)/2, rep(mutation_probability / 13, 13)))
  r_eta = sample(c(metadata[parents[1],xgb_eta],
                   metadata[parents[2],xgb_eta],
                   seq(0.01, 0.3, 0.005)), 1, prob=c((1-mutation_probability)/2, (1-mutation_probability)/2, rep(mutation_probability / 59, 59)))
  r_rounds = sample(c(metadata[parents[1],xgb_rounds],
                      metadata[parents[2],xgb_rounds],
                      50:500), 1, prob=c((1-mutation_probability)/2, (1-mutation_probability)/2, rep(mutation_probability / 451, 451)))
  r_seed = sample(c(metadata[parents[1],xgb_seed],
                    metadata[parents[2],xgb_seed],
                    0:2000), 1, prob=c((1-mutation_probability)/2, (1-mutation_probability)/2, rep(mutation_probability / 2001, 2001)))
  output_metadata = selected_vars
  names(output_metadata) = paste("Variable", 1:10)
  output_metadata = c(output_metadata,
                      ints_as_cats=inclusions[1],
                      prods=inclusions[2],
                      sums=inclusions[3],
                      diffs=inclusions[4],
                      quots=inclusions[5],
                      probs=inclusions[6],
                      num_probs=inclusions[7],
                      rare_cutoff=rare_cutoff,
                      xgb_depth=r_max_depth,
                      xgb_subsample=r_subsample,
                      xgb_colsample=r_colsample,
                      xgb_eta=r_eta,
                      xgb_rounds=r_rounds,
                      xgb_seed=r_seed)
  print.noquote(output_metadata)
  temp_dt = subset(all_dt, select=c("ID", "target", "train_test", "basesplit", selected_vars))
  
  # Treat integers as categoricals?
  if(inclusions[1]) {
    selected_types[selected_types=="int"] = "cat"
  } else {
    selected_types[selected_types=="int"] = "num"
  }
  
  # Now count the number of categoricals and numerics
  cats = sum(selected_types=="cat")
  catvars = names(selected_types[selected_types=="cat"])
  nums = sum(selected_types=="num")
  numvars = names(selected_types[selected_types=="num"])
  
  # 2-way products of numerics
  if(nums > 1 & inclusions[2]) {
    for (v in numvars) {
      for (w in numvars) {
        if (which(numvars==v) < which(numvars==w)) {
          temp_dt[,paste(v, "x", w, sep="") := temp_dt[,get(v)] * temp_dt[,get(w)]]
        }
      }
    }
  }
  
  # 2-way sums of numerics
  if(nums > 1 & inclusions[3]) {
    for (v in numvars) {
      for (w in numvars) {
        if (which(numvars==v) < which(numvars==w)) {
          temp_dt[,paste(v, "plus", w, sep="") := temp_dt[,get(v)] + temp_dt[,get(w)]]
        }
      }
    }
  }
  
  # 2-way differences of numerics
  if(nums > 1 & inclusions[4]) {
    for (v in numvars) {
      for (w in numvars) {
        if (which(numvars==v) < which(numvars==w)) {
          temp_dt[,paste(v, "minus", w, sep="") := temp_dt[,get(v)] - temp_dt[,get(w)]]
        }
      }
    }
  }
  
  # 2-way quotients of numerics
  if(nums > 1 & inclusions[5]) {
    for (v in numvars) {
      for (w in numvars) {
        if(v!=w) {
          quotvar = paste(v, "by", w, sep="")
          temp_dt[,(quotvar) := temp_dt[,get(v)] / temp_dt[,get(w)]]
          temp_dt[get(quotvar)==Inf, (quotvar) := NA]
        }
      }
    }
  }
  
  # Rare value cut off on categoricals
  if(cats > 0) {
    for (v in catvars) {
      raretable = table(temp_dt[,get(v)])
      raretable = raretable[raretable < rare_cutoff]
      temp_dt[as.character(get(v)) %in% names(raretable), (v) := NA]
    }
  }
  
  # y probabilities of categoricals (we do this on splits to avoid leakage)
  if(cats > 0 & inclusions[6]) {
    for (v in catvars) {
      probvar = paste("prob_", v, sep="")
      # fill trainB and test with the trainA probabilities
      probtab = table(temp_dt[basesplit=="trainA",get(v)], temp_dt[basesplit=="trainA",target])
      probtab = (probtab / cbind(rowSums(probtab), rowSums(probtab)))[,2]
      temp_dt[basesplit=="trainB" | basesplit=="test", (probvar) := probtab[as.character(all_dt[basesplit=="trainB" | basesplit=="test",get(v)])]]
      # fill trainA with the trainB probabilities
      probtab = table(temp_dt[basesplit=="trainB",get(v)], temp_dt[basesplit=="trainB",target])
      probtab = (probtab / cbind(rowSums(probtab), rowSums(probtab)))[,2]
      temp_dt[basesplit=="trainA", (probvar) := probtab[as.character(temp_dt[basesplit=="trainA",get(v)])]]
    }
  }
  
  # bin numerics and take y probabilities of numeric bins (on splits)
  if(nums > 0 & inclusions[7]) {
    howmanynumvars = length(numvars)
    i = 0
    for (v in numvars) {
      i = i + 1
      cat(paste("Calculating num probs for", v, "(", i, "of", howmanynumvars, ") ..."))
      probvar = paste("prob_", v, sep="")
      probs_dt[,(v) := round(get(v), 1)]
      # fill trainB and test with the trainA probabilities
      probtab = table(probs_dt[basesplit=="trainA",get(v)])
      probtab = probtab / nrow(probs_dt[basesplit=="trainA",])
      probs_dt[basesplit=="trainB" | basesplit=="test", (probvar) := probtab[as.character(probs_dt[basesplit=="trainB" | basesplit=="test",get(v)])]]
      # fill trainA with the trainB probabilities
      probtab = table(probs_dt[basesplit=="trainB",get(v)])
      probtab = probtab / nrow(probs_dt[basesplit=="trainB",])
      probs_dt[basesplit=="trainA", (probvar) := probtab[as.character(probs_dt[basesplit=="trainA",get(v)])]]
    }
  }
  
  # Convert all categoricals to integer values (XGBoost should be able to handle this okay, especially with rare value cutoff)
  if(cats > 0) {
    for (v in catvars) {
      set(temp_dt, j=v, value=as.numeric(as.factor(temp_dt[,get(v)])))
    }
  }
  
  # Now set up our DMatrix objects
  trainingvars = setdiff(names(temp_dt), c("ID", "target", "train_test", "basesplit"))
  dtrainA = xgb.DMatrix(as.matrix(subset(temp_dt, select=trainingvars, subset=basesplit=="trainA")), label=temp_dt[basesplit=="trainA",target], missing=NA)
  dtrainB = xgb.DMatrix(as.matrix(subset(temp_dt, select=trainingvars, subset=basesplit=="trainB")), label=temp_dt[basesplit=="trainB",target], missing=NA)
  dtest = xgb.DMatrix(as.matrix(subset(temp_dt, select=trainingvars, subset=basesplit=="test")), missing=NA)
  # Set XGB parameters
  param = list(objective = "binary:logistic", eta = r_eta,
               max_depth = r_max_depth, subsample = r_subsample, colsample_bytree=r_colsample,
               metrics = "logloss")
  # Train model A and make predictions on B and test
  set.seed(r_seed)
  bstA = xgboost(metrics = "logloss", nrounds=r_rounds, params=param, verbose=0, data=dtrainA)
  temp_dt[basesplit=="trainB", predictions := predict(bstA, dtrainB)]
  temp_dt[basesplit=="test", predictions := predict(bstA, dtest)]
  # Train model B and make predictions on A
  set.seed(r_seed)
  bstB = xgboost(metrics = "logloss", nrounds=r_rounds, params=param, verbose=0, data=dtrainB)
  temp_dt[basesplit=="trainA", predictions := predict(bstB, dtrainA)]
  
  # Estimate trainA and trainB error
  trainA_logloss = quicklogloss(temp_dt[basesplit=="trainA", predictions], temp_dt[basesplit=="trainA", target])
  trainB_logloss = quicklogloss(temp_dt[basesplit=="trainB", predictions], temp_dt[basesplit=="trainB", target])
  print.noquote(paste("Mean loss:", round(mean(c(trainA_logloss, trainB_logloss)), 4),
                      "| trainA loss:", round(trainA_logloss, 4),
                      "| trainB loss:", round(trainB_logloss, 4)))
  
  # Output information to file
  output_metadata = c(filename=predictions_filename, max_logloss = max(c(trainA_logloss, trainB_logloss)),
                      A_logloss=trainA_logloss, B_logloss=trainB_logloss, output_metadata)

  write.table(t(output_metadata), metadata_file, quote=F, row.names=F, sep=",", append=T, col.names=F)
  
  # Output predictions to file
  write.csv(temp_dt[,predictions], predictions_filename, quote=F, row.names=F)
  
  # Tidy up the bigger objects before going again
  rm(list=c("temp_dt", "dtrainA", "dtrainB", "dtest"))
  gc()
}
