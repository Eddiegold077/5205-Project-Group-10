---
title: "R Code For Obtaining the Cleaned Dataset"
author: "Group 10: Xinyu Chen, Jiaying Feng, Yujia Ge, Jintong He, Zhiyao Lin"
output: html_document
---

## Data preparation
```{r}
library(skimr)
heart= read.csv('/Users/jintonghe/Desktop/5205/Term\ Project/heart.csv')  
# replace the file path with your own 
skim(heart)
```
```{r}
value_counts <- table(heart$HeartDisease)
value_counts 
```
Description: There are 508 people with heart failure and 410 people without heart failure.
```{r}
labels <- c('0', '1')
colors <- c('cyan', 'pink')
pie(value_counts, labels = labels, col = colors, main = "Heart Disease", clockwise = TRUE)

legend("topright", legend = c("Normal", "Heart disease"), fill = colors)
```
```{r}
suppressPackageStartupMessages(library(dplyr))
heart <- heart %>% mutate_if(is.character, as.factor)
```

In order to overview the data, we drew histogram and got the distribution of each variable. Our main research subjects are individuals aged between 40 and 60 years old. The resting BP and max HR follow normal distribution with mean 132 and 136 respectivily.
```{r}
# draw hist for all numeric columns except 'HeartDisease' to find the distribution 
numeric_cols <- sapply(heart[,-12], is.numeric)
numeric_col_names <- names(heart[,-12])[numeric_cols]
par(mfrow = c(ceiling(sqrt(length(numeric_col_names))), ceiling(sqrt(length(numeric_col_names)))))
for (col in numeric_col_names) {
  hist(heart[[col]], main = col, xlab = "Value", ylab = "Frequency", col = "pink")
}
```

The dataset contains the following factor variables and their respective observations:

Gender (Gender):  
Female (F): 193  
Male (M): 725

Cardiovascular Conditions (Condition):  
Asymptomatic (ASY): 496  
Atrial Tachycardia (ATA): 173
Non-anginal Pain (NAP): 203  
Typical Angina (TA): 46

Left Ventricular Hypertrophy (LVH):  
Present (Y): 188  
Absent (N): 552

ST Segment Abnormality (ST):  
Upward Sloping (Up): 395   
Flat (Flat): 460
Downward Sloping (Down): 63

T-wave Abnormality (TWAVE):  
Present (Y): 371   
Absent (N): 547
```{r}
# draw hist for all factor columns to find the distribution 
factor_cols <- sapply(heart, is.factor)
factor_col_names <- names(heart)[factor_cols]
par(mfrow = c(ceiling(sqrt(length(factor_col_names))), ceiling(sqrt(length(factor_col_names)))))
for (col in factor_col_names) {
  sex_table <- table(heart[[col]])
  barplot(sex_table, main = col, xlab = "type", ylab = "Frequency", col = "pink")
}
```
```{r}
# Split Data
heart <- heart %>% 
  mutate_if(is.factor, as.integer)
suppressPackageStartupMessages(library(caret))
set.seed(1031)
split = createDataPartition(y=heart$HeartDisease,p = 0.7,list = F,groups = 100)
train_heart = heart[split,]
test_heart = heart[-split,]
str(train_heart)
```

## Feature Selection
Predictor variables were screened based on their relationship to the outcome variable (correlation) and their relationship to other predictor variables (redundancy), thus retaining relevant and non-redundant variables.
The method for screening predictors by examining their bivariate correlations has been conducted and correlation heat map has been constructed.
```{r}
#Bivariate Filter
library(ggcorrplot)
numeric_data <- train_heart[sapply(train_heart, is.numeric)]
str(numeric_data)
numeric_cor <- cor(numeric_data)
ggcorrplot(numeric_cor,
           method = 'square',
           type = 'lower',
           show.diag = FALSE,  # do not show diagnoal
           colors = c('#e9a3c9', '#f7f7f7', '#a1d76a'))
```


Correlation and redundancy of all predictors have been examined, since predictors that are found to be relevant and non-redundant based on binary correlations may be found to be redundant when considering the effects of other predictors. To determine if this is the case, variance inflation factors(VIF) has been used.
```{r}
# Multivariate Filter
model <- lm(HeartDisease ~ ., data = numeric_data)
library(broom)
summary(model) |>
  tidy()
```
```{r}
# Assess the threat of multicollinearity in a linear regression by computing the Variance Inflating Factor (VIF).
suppressPackageStartupMessages(library(car))
vif(model)
```
```{r}
# VIF visualization
data.frame(Predictor = names(vif(model)), VIF = vif(model)) |>
  ggplot(aes(x=VIF, y = reorder(Predictor, VIF), fill=VIF))+
  geom_col()+
  geom_vline(xintercept=5, color = 'gray', linewidth = 1.5)+
  geom_vline(xintercept = 10, color = 'red', linewidth = 1.5)+
  scale_fill_gradient(low = '#fff7bc', high = '#d95f0e')+
  scale_y_discrete(name = "Predictor")+
  scale_x_continuous(breaks = seq(5,30,5))+
  theme_classic()
```

All numerical features remain after the process, since VIF < 5.

Stepwise Variable Selection has been conducted to detect the best feature groups. After the Hybrid Stepwise Visualization,4 numerical variables remained.
```{r}
#Subset Selection
#Stepwise Variable Selection
start_mod = lm(HeartDisease~1,data=numeric_data)
empty_mod = lm(HeartDisease~1,data=numeric_data)
full_mod = lm(HeartDisease~.,data=numeric_data)
hybridStepwise = step(start_mod,
                      scope=list(upper=full_mod,lower=empty_mod),
                      direction='both')
```

```{r}
summary(hybridStepwise)
```
```{r}
# Hybrid Stepwise Visualization
library(dplyr)
hybridStepwise$anova |> 
  mutate(step_number = as.integer(rownames(hybridStepwise$anova))-1) |>
  mutate(Step = as.character(Step))|>
  ggplot(aes(x = reorder(Step,X = step_number), y = AIC))+
  geom_point(color = 'darkgreen', size = 2) + 
  scale_x_discrete(name = 'Variable Added or Dropped')+
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.9, hjust=0.9))
```

Lasso Regression has been conducted.With the one-standard-error rule, 4 coefficients have not been forced to exactly zero
```{r}
#Shrinkage
#Lasso Regression
suppressPackageStartupMessages(library(glmnet))
x <- model.matrix(HeartDisease ~ . - 1, data = numeric_data)
y <- numeric_data$HeartDisease
set.seed(1031)
cv_lasso = cv.glmnet(x = x, 
                     y = y, 
                     alpha = 1,
                     type.measure = 'mse')
cv_lasso
```
```{r}
coef(cv_lasso, s = cv_lasso$lambda.1se) |>
  round(6)
```
```{r}
# Obtain the cleaned dataset after lasso regression
lambda = cv_lasso$lambda.min
coefficients = coef(cv_lasso, s = lambda)
non_zero_coef = coefficients[coefficients[,1] !=0, 1, drop = F]
selected_variables = c(rownames(non_zero_coef)[-1], 'HeartDisease')
cleaned_data = heart[, selected_variables]
```
```{r}
# Output the cleaned data
write.csv(cleaned_data, '/Users/jintonghe/Desktop/5205/Term\ Project/cleaned_data.csv', row.names = F)
# replace the file path with your own
```
## Principal Components Analysis
```{r}
# Drop outcome variable, **HeartDisease** for train and test sets
train = train_heart[,1:11]
test = test_heart[,1:11]
```
```{r}
# Bartlett's test of sphericity
suppressPackageStartupMessages(library(psych))
cortest.bartlett(cor(train),n = nrow(train))
```
```{r}
# KMO measure of sampling adequacy (MSA)
KMO(cor(train)) 
``` 

```{r}
# Scree plot
library(FactoMineR)
pca_facto = PCA(train, graph = F)
suppressPackageStartupMessages(library(factoextra))
fviz_eig(pca_facto, ncp=11, addlabels = T)
```

The graph of eigen values for each component is shown as above. It can be noticed that sudden changes (also known as the elbow) appear around 2, 3, and 7 dimensions, indicating that the ideal number of components will be 2, 3, or 7.
```{r}
# eigenvalue criterion
pca_facto$eig
```
```{r}
# select components with eigen value greater than 1
pca_facto$eig[pca_facto$eig[,'eigenvalue']>1,]
```

```{r}
# Parallel analysis
fa.parallel(train,fa='pc')
```

```{r}
pca_facto$eig
```
If we would like to retain over 70% of the total variance, we would select at least the first 6 principal components.

Summary of test results:
The results from each method differ widely:

Scree Plot: 2, 3, or 7 components  
  
Eigen Value: 4 components   
  
Parallel Analysis: 3 components   
  
However, the use of any fewer than 6 components would explain less than 70% of the original data. So, we go with a seven-component structure suggested by the Scree plot.

```{r}
# Describe components
pca_facto = PCA(train,scale.unit = T,ncp = 7,graph = F)
pca_facto$var$contrib %>%
  round(2)
```
```{r}
# Contributions of each variable to each component are charted out below.
suppressPackageStartupMessages(library(gridExtra))
charts = lapply(1:7,FUN = function(x) fviz_contrib(pca_facto,choice = 'var',axes = x,title=paste('Dim',x)))
grid.arrange(grobs = charts,ncol=3,nrow=3)
```

```{r}
# examine the relationships between variables
fviz_pca_var(X = pca_facto,col.var = 'contrib',gradient.cols = c('red'),col.circle = 'steelblue',repel = T)
```

The plot charts the first two most important components, Dim1 and Dim2 which explains 40.4% of the total variance. Angle between a variable and component reflects strength of relationship (smaller the angle, stronger the relationship) and the color indicates the contribution of the variable to the first two components. The picture is helpful but one must bear in mind that it only represents the first two components (44% of variance). All seven components represent 80% of variance.

```{r}
#Apply component structure
trainComponents = pca_facto$ind$coord
testComponents = predict(pca_facto,newdata=test)$coord

trainComponents = cbind(trainComponents,quality = train$HeartDisease)
testComponents = cbind(testComponents,quality = test$HeartDisease)
```

## Factor Analysis
```{r}
# KMO measure of sampling adequacy (MSA)
KMO(cor(train)) 
```
```{r}
# Determine number of factors through parallel analysis
fa.parallel(heart, fa = 'fa', fm = 'pa')
```
```{r}
# Extract communalities
result = suppressMessages(fa(r = heart, nfactors = 4, fm = 'pa', rotate = 'none'))
result$Vaccounted
```
```{r}
# Show extracted communalities
data.frame(communality = result$communality) 
```
Although the MSA (mean square error) value is greater than 0.5, indicating that the dataset is suitable for factor analysis, the fact that the communalities of most variables are lower than 0.5 suggests that the shared variance among these variables is insufficient to support the feasibility of factor analysis.  

## Conclusion  
After feature selection, principal components analysis, and factor analysis, we have decided to adopt the dataset obtained through feature selection to better meet the analytical needs. Through feature selection, we can identify the most relevant variables with significant communalities, thereby enhancing the interpretability and effectiveness of the analysis results.


