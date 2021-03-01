# Poisson regression for DPVI females

library(Epi)
library(data.table)

path_to_data = path

## Read data
# Females
female.data = fread(paste(path_to_data,"female_aggregated_data.csv", sep=""), 
                                header = T, stringsAsFactors = T, sep = ',')
female.data = as.data.frame(female.data)

# Males
male.data = fread(paste(path_to_data,"male_aggregated_data.csv", sep=""), 
                                header = T, stringsAsFactors = T, sep = ',')
male.data = as.data.frame(male.data)

## Relevel features
for (key in colnames(female.data)){
  if(is.factor(female.data[, key])){
    female.data[, key] = factor(female.data[, key], level.list[[key]])
  }
}
## Relevel features
for (key in colnames(male.data)){
  if(is.factor(male.data[, key])){
    male.data[, key] = factor(male.data[, key], level.list[[key]])
  }
}

############
# Alcohol, multivariate models
  
female_formula = as.formula(
  "ep~Relevel(age.cat,list(1:2,3,4,5,6,7:9))+
    I(as.numeric(per.cat))+(C10AA.DDD>0)+(G03.DDD>0)+
    DM.type+cut(lex.dur,c(0,0.5,1:5,10,12.5,Inf))+
    .i.cancer+factor(shp)+offset(log(lex.dur+1e-6))"
)

male_formula = as.formula(
  "ep~Relevel(age.cat,list(1:2,3,4,5,6,7:9))+
    I(as.numeric(per.cat))+(C10AA.DDD>0)+
    DM.type+cut(lex.dur,c(0,0.5,1:5,10,12.5,Inf))+
    .i.cancer+factor(shp)+offset(log(lex.dur+1e-6))"
)

############
# Females
poisson.fit <- function(formula, data, indices) {
  d <- data[indices,] # allows boot to select sample
  fit <- glm(formula, data=d, y=FALSE, model=FALSE, family=poisson)
  return(summary(fit)$coefficients[, c(1,4)])
}

library(boot)
female.coefs = boot(data=female.data, statistic=poisson.fit, R=100, formula=female_formula)
write.csv(female.coefs$t, file="female_bootstrap.csv")

male.coefs = boot(data=male.data, statistic=poisson.fit, R=100, formula=male_formula)
write.csv(male.coefs$t, file="male_bootstrap.csv")

write.csv(female.coef$t0, file="female_header.csv")
write.csv(male.coef$t0, file="male_header.csv")
