library(readxl)
library(bonsai)
library(tidymodels)
library(tidyverse)
library(finetune)
library(rsample)
library(ggplot2)
theme_set(theme_minimal())
#remotes::install_github("curso-r/treesnip")
#remotes::install_github("curso-r/treesnip@catboost")
library(rsample)
library(recipes)
library(yardstick)
library(treesnip)
library(doParallel)

Augment  <- read_excel("D:/My Scripts/Github/Poverty Estimation/Classifying Images/VGG16/Augment Model Extracted.xlsx")
Base  <- read_excel("D:/My Scripts/Github/Poverty Estimation/Classifying Images/VGG16/Base Model Extracted.xlsx")
BatchNorm  <- read_excel("D:/My Scripts/Github/Poverty Estimation/Classifying Images/VGG16/BatchNorm Model Extracted.xlsx")
Dropout  <- read_excel("D:/My Scripts/Github/Poverty Estimation/Classifying Images/VGG16/Dropout Model Extracted.xlsx")
L1  <- read_excel("D:/My Scripts/Github/Poverty Estimation/Classifying Images/VGG16/L1 Model Extracted.xlsx")
L1L2  <- read_excel("D:/My Scripts/Github/Poverty Estimation/Classifying Images/VGG16/L1L2 Model Extracted.xlsx")
L2  <- read_excel("D:/My Scripts/Github/Poverty Estimation/Classifying Images/VGG16/L2 Model Extracted.xlsx")


data_prep = function(data, k=10, repeats=5) {
  data[,1] = NULL
  colnames(data)[33] = 'y'
  data1 = data %>%
    vfold_cv(v=k, repeats = repeats)
  output = list(data=data, resample=data1)
  return(output)
}

Augment_cv = data_prep(Augment)
Base_cv = data_prep(Base)
BatchNorm_cv = data_prep(BatchNorm)
Dropout_cv = data_prep(Dropout)
L1_cv = data_prep(L1)
L1L2_cv = data_prep(L1L2)
L2_cv = data_prep(L2)

## 
# speed up computation with parallel processing
all_cores <- parallel::detectCores(logical = FALSE)

# specifying models
lightgbm_model<-
  parsnip::boost_tree(
    mode = "regression",
    trees = 1000,
    min_n = tune(),
    tree_depth = tune(),
  ) %>%
  set_engine("lightgbm",verbose=-1)

svm = svm_rbf(mode = 'regression', cost = tune(), rbf_sigma = tune())

xgboost = boost_tree(mode = 'regression', trees = 1000, min_n = tune(), tree_depth = tune())

rf = rand_forest(mode = 'regression', engine = 'randomForest', mtry = tune(), trees = 1000, min_n = tune())

my_wf = workflow_set(
  preproc = list(my_recipe),
  models = list('lightgbm'=lightgbm_model, 'svm'=svm, 'xgboost'=xgboost, 'rf'=rf),
  cross = T) 
my_wf

race_ctrl <-
  control_race(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
  )

race_res %>% 
  rank_results() 

race_res %>% 
  rank_results() %>%
  filter(wflow_id=="recipe_rf")

train_model = function(mydata, number_of_grid=100) {
  registerDoParallel(cores = (all_cores-1))
  my_recipe = recipe(y~., data = mydata$data) %>%
    step_nzv(all_predictors())
  my_wf = workflow_set(
    preproc = list(my_recipe),
    models = list('lightgbm'=lightgbm_model, 'svm'=svm, 'xgboost'=xgboost, 'rf'=rf),
    cross = T)
  race_res = my_wf %>% workflow_map('tune_race_anova', seed = 135, resamples = mydata$resample,
                                    grid=number_of_grid, control=race_ctrl)
  return(race_res)
}

augmen_res = train_model(Augment_cv)
base_res = train_model(Base_cv)
batchnorm_res = train_model(BatchNorm_cv)
dropout_res = train_model(Dropout_cv)
l1_res = train_model(L1_cv)
l1l2_res = train_model(L1L2_cv)
l2_res = train_model(L2_cv)

l1_res %>% 
  rank_results() %>%
  filter(wflow_id=='recipe_rf') %>%
  slice_head(n=2)

extract_each_model = function(tidy_result, data_name) {
  wflow = c('recipe_rf', 'recipe_lightgbm', 'recipe_svm', 'recipe_xgboost')
  best_model_each = data.frame(matrix(data = NA, nrow = 2*length(wflow), ncol = NCOL(rank_results(tidy_result))))
  colnames(best_model_each) = colnames(rank_results(tidy_result))
  for (i in 1:length(wflow)) {
    best_model_each[c(2*i-1,2*i),] = tidy_result %>%
      rank_results() %>%
      filter(wflow_id==wflow[i]) %>%
      slice_head(n=2)
  }
  best_model_each$data = data_name
  return(arrange(best_model_each, rank))
}
augment_extracted = extract_each_model(augmen_res, 'Augment')
base_extracted = extract_each_model(base_res, 'Base')
batchnorm_extracted = extract_each_model(batchnorm_res, 'Batchnorm')
dropout_extracted = extract_each_model(dropout_res, ' Dropout')
l1_extracted = extract_each_model(l1_res, 'L1')
l1l2_extracted = extract_each_model(l1l2_res, 'L1L2')
l2_extracted = extract_each_model(l2_res, 'L2')
all_data_extracted = rbind.data.frame(augment_extracted, base_extracted, batchnorm_extracted, dropout_extracted,
                                      l1_extracted, l1l2_extracted, l2_extracted)
str(all_data_extracted)
all_data_extracted$model = as.factor(all_data_extracted$model)
all_data_extracted$data = as.factor(all_data_extracted$data)

all_data_extracted %>%
  filter(.metric %in% "rmse") %>%
  ggplot() +
  aes(x = wflow_id, y = data, fill = mean, label=round(mean,3)) +
  geom_tile(size = 1.2) +
  scale_fill_distiller(palette = "Oranges", direction = 1) +
  labs(x='Regression Model', y='Data Extracted from Images', fill='RMSE') +
  geom_text() +
  theme_minimal()

all_data_extracted %>%
  filter(.metric %in% "rsq") %>%
  ggplot() +
  aes(x = wflow_id, y = data, fill = mean, label=round(mean,3)) +
  geom_tile(size = 1.2) +
  scale_fill_distiller(palette = "Oranges", direction = -1) +
  labs(x='Regression Model', y='Data Extracted from Images', fill='Rsquared') +
  geom_text() +
  theme_minimal()

model_rank_rsq = data.frame(all_data_extracted %>%
                              filter(.metric %in% "rsq") %>%
                              arrange(-mean) %>%
                              select(wflow_id, data, mean, std_err),
                            'rank' = 1:(NROW(all_data_extracted)/2))
model_rank_rsq = model_rank_rsq %>% arrange(wflow_id, data)
  
model_rank_rmse = data.frame(all_data_extracted %>%
                              filter(.metric %in% "rmse") %>%
                              arrange(mean) %>%
                              select(wflow_id, data, mean, std_err),
                            'rank' = 1:(NROW(all_data_extracted)/2))
model_rank_rmse = model_rank_rmse %>% arrange(wflow_id, data)
all_model_rank = model_rank_rsq %>% left_join(model_rank_rmse, 
                             by = c('wflow_id', 'data'), 
                             suffix = c(".rsq",".rmse"))
all_model_rank$avg.rank = (all_model_rank$rank.rsq + all_model_rank$rank.rmse)/2
knitr::kable(head(all_model_rank), 'simple')
all_model_rank = all_model_rank %>% arrange(avg.rank)
ggplot(all_model_rank) +
  aes(x = wflow_id, y = data, fill = avg.rank, label=avg.rank) +
  geom_tile(size = 1.2) +
  geom_text() +
  scale_fill_distiller(palette = "Oranges", direction = 1) +
  labs(x='Regression Model', y='Data Extracted from Images', fill='Average Rank', caption = 'lower better') +
  theme_minimal() +
  theme(legend.position = 'top')

### create predictions
# model with lowest rmse
best_batchnorm_param = batchnorm_res %>%
  extract_workflow_set_result('recipe_svm') %>%
  select_best(metric='rmse')
best_batchnorm_res = batchnorm_res %>% 
  extract_workflow('recipe_svm') %>% 
  finalize_workflow(best_batchnorm_param) %>%
  fit(BatchNorm_cv$data) %>%
  predict(BatchNorm_cv$data)
best_rmse = unlist(batchnorm_res %>% 
                     rank_results() %>%
                     select(mean) %>%
                     slice(n=1))
MLmetrics::RMSE(unlist(best_batchnorm_res), BatchNorm_cv$data$y)
ggplot(data = NULL, aes(x=BatchNorm_cv$data$y, y=unlist(best_batchnorm_res))) +
  geom_point() + coord_obs_pred() + labs(x='Actual', y='Estimation') +
  geom_abline(slope = 1) + geom_abline(slope = 1, intercept = best_rmse, linetype='dashed') +
  geom_abline(slope = 1, intercept = -best_rmse, linetype='dashed')

# model with highest rsq also bet overall model
best_l1l2_param = l1l2_res %>%
  extract_workflow_set_result('recipe_rf') %>%
  select_best(metric='rsq')
best_l1l2_res = l1l2_res %>% 
  extract_workflow('recipe_rf') %>% 
  finalize_workflow(best_l1l2_param) %>%
  fit(L1L2_cv$data) %>%
  predict(L1L2_cv$data)
l1l2_res %>% 
  rank_results()
best_l1l2_metric = unlist(l1l2_res %>% 
                     rank_results() %>%
                     select(mean) %>%
                     slice(n=1))
ggplot(data = NULL, aes(x=L1L2_cv$data$y, y=unlist(best_l1l2_res))) +
  geom_point() + coord_obs_pred() + labs(x='Actual', y='Estimation')+
  geom_abline(slope = 1) + geom_abline(slope = 1, intercept = best_l1l2_metric, linetype='dashed') +
  geom_abline(slope = 1, intercept = -best_l1l2_metric, linetype='dashed')
