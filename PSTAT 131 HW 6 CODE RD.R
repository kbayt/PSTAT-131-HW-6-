# LOADING PACKAGES & DATA
library(ggplot2)
library(tidyverse)
library(tidymodels)
library(corrplot)
library(ggthemes)
tidymodels_prefer()
library(ISLR)
library(yardstick)
library(corrr)
library(discrim)
library(poissonreg)
library(klaR)
library(janitor)
library(randomForest)
library(vip)
library(rpart.plot)
library(xgboost)
library(ranger)
set.seed(4857)

## EXERCISE 1
# read in data
pokemon <- read.csv("C:\\Pokemon.csv")
# clean names
pokemon <- pokemon %>% clean_names
view(pokemon)
# filter out rarer types 
pokemon <- pokemon %>%
  filter(type_1 == "Bug" | type_1 == "Fire" |
           type_1 == "Grass" | type_1 == "Normal" |
           type_1 == "Water" | type_1 == "Psychic")
# convert type_1 and legendary to factors
pokemon$type_1 <- factor(pokemon$type_1)
pokemon$legendary <- factor(pokemon$legendary)
# inital split on data, stratify on type_1
pokemon_split <- initial_split(pokemon, prop = 0.7,
                               strata = type_1)
pokemon_train <- training(pokemon_split)
pokemon_test <- testing(pokemon_split)
# 318
dim(pokemon_train)
# 140
dim(pokemon_test)
# v-fold cross validation with v=5
pokemon_folds <- vfold_cv(pokemon_train, v=5, 
                          strata = type_1)
# recipe
pokemon_recipe <- recipe(type_1 ~ legendary +
                           generation + sp_atk +
                           attack + speed + defense +
                           hp + sp_def, data = pokemon_train) %>%
  step_dummy(legendary, generation) %>%
  step_normalize(all_predictors())

## EXERCISE 2
# correlation matrix using corrplot 
pokemon %>%
  select(where(is.numeric)) %>%
  select(-c(x, total)) %>%
  cor() %>%
  corrplot(type = 'lower', diag = FALSE)
#could remove total because will be positively correlated to verything summed in it
# could remove x - just ID
# point out relationships in write up

## EXERCISE 3
# set up decision tree model and workflow:
## create general decision treep specification
tree_spec <- decision_tree() %>%
  set_engine("rpart")
# create classification decision tree spec
pokemon_tree_spec <- tree_spec %>%
  set_mode("classification")
# set up workflow + tune cost complexity 
pokemon_tree_wf <- workflow() %>%
  add_model(pokemon_tree_spec %>% 
              set_args(cost_complexity = tune())) %>%
  add_recipe(pokemon_recipe)
# tune the hyper parameter cost complexity 
set.seed(4857)
param_grid <- grid_regular(cost_complexity(
  range=c(-3,-1)), levels = 10)

tune_res <- tune_grid(
  pokemon_tree_wf,
  resamples = pokemon_folds,
  grid = param_grid,
  metrics = metric_set(roc_auc)
)

# plot results
autoplot(tune_res)
# optimal roc_auc is around complexity of 0.2 
# peforms better with smaler complexity values
# should result in complex model 

## QUESTION 4
# roc_auc of best performing pruned decision tree
best_complexity <- select_best(tune_res)

## QUESTION 5
# fit and visualize best-performing pruned deciiosn
# tree with training set
pokemon_tree_final <- finalize_workflow(
  pokemon_tree_wf, best_complexity)

pokemon_tree_final_fit <- fit(pokemon_tree_final,
                              data = pokemon_train)
pokemon_tree_final_fit %>%
  extract_fit_engine() %>%
  rpart.plot()

## EXERCISE 5 (CONT)
# do greater than 8 trees and make minimal greater too 
# set up RF model and workflow 
rf_spec <- rand_forest() %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_wkflow <- workflow() %>%
  add_model(rf_spec %>%
              set_args(mtry = tune(), trees = tune(),
                       min_n = tune())) %>%
  add_recipe(pokemon_recipe)


set.seed(4857)
multi_param_grid <- grid_regular(
  mtry(range = c(1,8)), trees(range(1,200)),
  min_n(range(1,30)), levels = 8)

multi_tune_res <- tune_grid(
  rf_wkflow,
  resamples = pokemon_folds,
  grid = multi_param_grid,
  metrics = metric_set(roc_auc))
# model with mtry=8 would be bagging model bc
# would have all predictors
# cant be > 8 bc dont have that many predictos
# cant be < 1 bc cant have negative predictors 

# EXERCISE 6
autoplot(multi_tune_res)
# look into best after fixing metric 
# 7 and 8 trees seemed best 
# best is 6 preditrs with highest minimal, and 
# 6 trees


# EXERCISE 7
# best performing model 
best_model <- select_best(multi_tune_res, 
                          metric = "roc_auc")

# EXERCISE 8
# variable importance plot using vip() w best model
# fit to training et 
best_model_final <- finalize_workflow(rf_wkflow,
                                      best_model)
best_model_final_fit <- fit(best_model_final,
                            data = pokemon_train)
# vip() plot
best_model_final_fit %>% extract_fit_engine() %>% vip()
# works, need to talk about resutls 

## EXERCISE 9
# Set up boosted tree model and worflow 
boost_spec <- boost_tree() %>%
  set_engine("xgboost") %>%
  set_mode("classification")
boost_wkflow <- workflow() %>%
  add_model(boost_spec %>%
              set_args(trees = tune())) %>%
  add_recipe(pokemon_recipe)
# create regular grid with 10 levels
trees_grid <- grid_regular(trees(range=c(10,2000)), 
                           levels = 10)

tree_tune_res <- tune_grid(
  boost_wkflow,
  resamples = pokemon_folds,
  grid = trees_grid,
  metrics = metric_set(roc_auc))
# plot results
autoplot(tree_tune_res)
# the best value is at minimal trees (10)

# best performing model
best_boost_trees <- select_best(tree_tune_res)
best_boost_model <- finalize_workflow(boost_wkflow,
                                      best_boost_trees)
best_boost_model_fit <- fit(best_boost_model,
                            data = pokemon_train)

## EXERCISE 10
# pruned tree, rf, and boosted tree
pruned_tree_roc_auc <- collect_metrics(tune_res) %>% arrange(-mean) %>% head()
rf_roc_auc <- collect_metrics(multi_tune_res) %>% arrange(-mean) %>% head()
boost_roc_auc <- collect_metrics(tree_tune_res) %>% arrange(-mean) %>% head()

roc_auc_vals <- c(pruned_tree_roc_auc$mean[1], rf_roc_auc$mean[1], boost_roc_auc$mean[1])
models <- c("Pruned Tree", "Random Forest", "Boosted Tree")
total_res <- tibble(roc_auc = roc_auc_vals,
                    models = models)
total_res
best_model_to_fit <- select_best(total_res)




summary(pokemon_tree_final_fit)  # pruned tree 
best_model_final_fit #rf 
best_boost_model_fit #boosted trees
# get values
pokemon_pruned_tree_res <-augment(pokemon_tree_final_fit,
                                  new_data = pokemon_train) %>%
  roc_auc(truth = type_1, estimate = c(.pred_Bug, 
                                       .pred_Fire, .pred_Grass, .pred_Normal,
                                       .pred_Psychic, .pred_Water))

pokemon_rf_res <-augment(best_model_final_fit,
                         new_data = pokemon_train) %>%
  roc_auc(truth = type_1, estimate = c(.pred_Bug, 
                                       .pred_Fire, .pred_Grass, .pred_Normal,
                                       .pred_Psychic, .pred_Water))
pokemon_boost_tree_res <- augment(best_boost_model_fit,
                                  new_data = pokemon_train) %>%
  roc_auc(truth = type_1, estimate = c(.pred_Bug, 
                                       .pred_Fire, .pred_Grass, .pred_Normal,
                                       .pred_Psychic, .pred_Water))

# create list of roc vals
roc_auc_vals <- c(pokemon_pruned_tree_res$.estimate,
                  pokemon_rf_res$.estimate,
                  pokemon_boost_tree_res$.estimate)
models <- c("Pruned Tree", "Random Forest",
            "Boosted Tree")

total_res <- tibble(roc_auc = roc_auc_vals,
                    models = models)
total_res %>%
  arrange(-roc_auc)

best_overall_train <- select_best(total_res, metric = "roc_auc")
