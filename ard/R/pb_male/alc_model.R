library(Epi)
library(data.table)

applyAnalysis <- function(eps, repetition){
  # Read synthetic
  fname_extend = paste(eps, "_", repetition, ".csv", sep="")
  male.synthetic.data = fread(paste(path_to_data,"/decoded/male_data_",fname_extend, sep=""), 
                                header = T, stringsAsFactors = T, sep = ',')
  male.synthetic.data = as.data.frame(male.synthetic.data)
  # Encode data for regression
  
  
  #Females
  male.synthetic.data$age.cat = cut(male.synthetic.data$age, c(-Inf,seq(10, 80, by=10),Inf),
                                      labels=level.list$age.cat)
  male.synthetic.data$per.cat = cut(male.synthetic.data$per, c(-Inf,seq(1997, 2012, by=1),Inf), 
                                      labels=level.list$per.cat)
  
  ## Relevel features
  for (key in colnames(male.synthetic.data)){
    if(is.factor(male.synthetic.data[, key])){
      male.synthetic.data[, key] = factor(male.synthetic.data[, key], level.list[[key]])
    }
  }
  
  ############
  # Alcohol, multivariate models
  
  male_formula = as.formula(
    "ep~Relevel(age.cat,list(1:2,3,4,5,6,7:9))+
    I(as.numeric(per.cat))+(C10AA.DDD>0)+
    DM.type+cut(lex.dur,c(0,0.5,1:5,10,12.5,Inf))+
    .i.cancer+factor(shp)+offset(log(lex.dur+1e-6))"
  )

  ############
  # Synthetic data
  ############
  # Females
  crng.alc.m1.f.syn=glm(male_formula, family=poisson, data=male.synthetic.data, y=FALSE, model=FALSE)  
  # Write results
  write.csv(crng.alc.m1.f.syn$coefficients, paste('./pb_csvs/male_coef_matrix_pb_',fname_extend,sep=""))
  write.csv(coef(summary(crng.alc.m1.f.syn))[,4], paste('./pb_csvs/male_p_value_matrix_pb_',fname_extend,sep=""))
  write.csv(coef(summary(crng.alc.m1.f.syn))[,2], paste('./pb_csvs/male_std_matrix_pb_',fname_extend,sep=""))
}

args = commandArgs(trailingOnly=TRUE)
eps = args[1]
nruns = as.integer(args[2])
print(eps)

for (repetition in 0:(nruns-1)){
  print(repetition)
  applyAnalysis(eps, repetition)
}
