library(mlbench)
library(doParallel)
doParallel::registerDoParallel(cores = 8)
data("PimaIndiansDiabetes2")
df <- PimaIndiansDiabetes2[ , -9]
diabetes <- PimaIndiansDiabetes2[ , 9]
y <- ifelse(diabetes=="pos", 1, 0)
df

# Checking the proportion of the two classes for the target variable, 
prop.pos <- sum(diabetes == "pos")/nrow(df)
prop.neg <- sum(diabetes == "neg")/nrow(df)
data.frame('Proporion of Positive Class' = prop.pos,
                 'Proportion of Negative Class' = prop.neg))

# Inspecting the structure of the dataset, and extracting basic descriptive statistics for all independent variables
library(modelsummary)
datasummary_skim(df, output = "DT")

# Data Visualization
library(reshape2)
meltedf <- melt(df)

library(ggplot2)
ggplot(meltedf, aes(x = value)) + 
  geom_histogram(bins = 9, color = "white",
                 fill = "#8A7FCD",
                 size = 0.5, alpha = 0.60) +
  facet_wrap(~variable, scales = "free") +
  theme_minimal() +
  labs(title = "Histograms Grid", x = "Value", y = "Frequency")

ggplot(meltedf, aes(x = value, fill = variable)) + 
  geom_density(adjust = 1, fill = "#8A7FCD", alpha = 0.60) +
  facet_wrap(~variable, scales = "free") +
  theme_minimal() +
  labs(title = "Density Distributions")

library(ggcorrplot)
correlation_matrix <- round(cor(data.frame(df, diabetes = y), use = "complete.obs", 
                                method = "spearman"), 3)
kable(correlation_matrix, caption = 'Correlations Matrix')

ggcorrplot(correlation_matrix, colors = c("yellow", "white", "#8A7FCD"), 
           show.diag = FALSE, lab = TRUE, lab_size = 2.8,
           digits = 3, tl.cex = 10, title = "Correlations Matrix")
eigen <- eigen(correlation_matrix)
kable(data.frame(eigenvalues = eigen$values)) 
kable(data.frame(kappa = kappa(correlation_matrix)))

# Checking for Near Zero Predictors
library(caret)
nzv <- nearZeroVar(df, saveMetrics = TRUE)
nzv

# Checking for duplicate records
duplicates <- df[duplicated(df), ]
sum(duplicates)

# Addressing the issue of outliers
ggplot(meltedf, aes(factor(variable), value)) +
  geom_boxplot(outlier.colour = '#8A7FCD', outlier.shape = 05, color = "#8A7FCD",
               fill = "white",) +
  facet_wrap(~variable, scale="free") +
  theme_minimal() +
  labs(title = "Predictors Boxplots", x = "Predictor variable", y = "Values") 

Pressure_Box <- boxplot(df$pressure, plot = FALSE)
Pressure_Outliers <- Pressure_Box$out
Pressure_Outliers <- subset(df$pressure, df$pressure <= 40)
Pressure_Outliers
df$pressure <- ifelse(df$pressure %in% Pressure_Outliers, NA, df$pressure)

Triceps_Box <- boxplot(df$triceps, plot = FALSE)
Triceps_Outliers <- Triceps_Box$out
Triceps_Outliers
df$triceps <- ifelse(df$triceps %in% Triceps_Outliers, NA, df$triceps)

# Addressing the issue of missing values
library(naniar)
library(tidyverse)
dim(df)
pct_miss(df)
pct_complete(df)
n_miss(df)
n_complete(df)
kable(miss_var_summary(df))
gg_miss_var(df, show_pct = TRUE)
vis_miss(df, sort_miss = TRUE) +
  scale_fill_manual(values = c("white", "#8A7FCD"))
vis_miss(arrange(df, insulin), sort_miss = TRUE) +
  scale_fill_manual(values = c("white", "#8A7FCD"))
mcar_test_df <- mcar_test(df)
mcar_test_df

# IMPUTING MISSING VALUES
library(mice)
imputed_df <- mice(df, m = 1, method = "rf",  seed = 123, maxit = 3, print = FALSE)
imputed_df <- complete(imputed_df, "all", include = TRUE)
imputed_df <- as.data.frame(imputed_df[2])
imputations <- data.frame(original_insulin = df$insulin,  
                          imputed_insulin = imputed_df$X1.insulin,
                          original_triceps = df$triceps,
                          imputed_triceps = imputed_df$X1.triceps)
imputations_long <- gather(imputations, method, value, factor_key = TRUE)

ggplot(imputations_long, aes(x = value, color = method)) +
  geom_density(aes(linetype = method), size = 1.2, na.rm = TRUE) +
  scale_color_manual(values = c("original_insulin" = "yellow", 
                                "imputed_insulin" = "#8A7FCD")) +
  scale_linetype_manual(values = c("original_insulin" = "solid", 
                                   "imputed_insulin" = "dotted")) +
  labs(x = "Value", y = "Density") +
  coord_cartesian(xlim = c(0, 900), ylim = c(0,0.0055)) +
  theme_minimal() +
  theme(legend.text = element_text(color = "#8A7FCD", face = "bold"))

ggplot(imputations_long, aes(x = value, color = method)) +
  geom_density(aes(linetype = method), size = 1.2, na.rm = TRUE) +
  scale_color_manual(values = c("original_triceps" = "yellow", 
                                "imputed_triceps" = "#8A7FCD")) +
  scale_linetype_manual(values = c("original_triceps" = "solid", 
                                   "imputed_triceps" = "dotted")) +
  labs(x = "Value", y = "Density") +
  coord_cartesian(xlim = c(0, 60), ylim = c(0,0.040)) +
  theme_minimal() +
  theme(legend.text = element_text(color = "#8A7FCD", face = "bold")) 

datasummary_skim(data.frame(imputed_df$X1.insulin, df$insulin, imputed_df$X1.triceps,
                            df$triceps), output = "DT")

ks.test(df$insulin, imputed_df$X1.insulin, alternative = "two.sided", 
        exact = TRUE,
        simulate.p.value = TRUE)

ks.test(df$triceps, imputed_df$X1.triceps, alternative = "two.sided", 
        exact = TRUE,
        simulate.p.value = TRUE)

# Preparing the datasets
library(tidymodels)
library(UBL)
set.seed(123)
df$diabetes <- diabetes
imputed_df$diabetes <- diabetes
colnames(imputed_df) <- colnames(df)
df_split <- initial_split(imputed_df, prop = 4/5, strata = diabetes)
train <- training(df_split)
test <- testing(df_split)
balanced_train <- AdasynClassif(diabetes~., train, beta = 1, k = 3, 
                                dist = "Euclidean")
nrow(subset(test, diabetes == "pos"))/nrow(test)
nrow(subset(train, diabetes == "pos"))/nrow(train)
nrow(subset(balanced_train, diabetes == "pos"))/nrow(balanced_train)
nrow(subset(balanced_train, diabetes == "neg"))/nrow(balanced_train)
folds <- vfold_cv(train, v = 5, strata = diabetes)
balanced_folds <- vfold_cv(balanced_train, v = 5, strata = diabetes)

# MODEL & EVALUATION PHASE
# Decision Tree
# Original/Unbalanced approach
library(rpart)
library(rpart.plot)
library(pROC)
set.seed(123)
decision_tree <- decision_tree(
                          cost_complexity = 0.02,
                          tree_depth = tune(),
                          min_n = tune()
                                        ) %>%
                          set_mode("classification") %>%
                          set_engine(engine = "rpart", 
                                     parms = list(split = "information"))

tree_recipe <- recipe(data = train, formula = diabetes~.)

tree_workflow <- workflow() %>%
                  add_recipe(tree_recipe) %>%
                  add_model(decision_tree) 
               
tree_parameters_grid <- grid_regular(parameters(decision_tree), levels = 10)

tree_tune_process <- tune_grid(tree_workflow,
                          resamples = folds,
                          grid = tree_parameters_grid,
                          metrics = metric_set(roc_auc))

autoplot(tree_tune_process)

tree_best_process <-  select_best(tree_tune_process, metric = 'roc_auc')

best_tree <- finalize_model(decision_tree, tree_best_process)
best_tree

resamples_fit <- fit_resamples(best_tree, diabetes ~ ., 
                               resamples = folds)
collect_metrics(resamples_fit)

tree_best_fit <- fit(best_tree, diabetes~., train)
tree_best_fit

tree_pred.classes <- predict(tree_best_fit, test, type = "class")
tree_pred.classes <- factor(unlist(tree_pred.classes))
tree_confusion_matrix <- confusionMatrix(data = tree_pred.classes, 
                                         reference = test$diabetes, 
                                         positive = "pos")

tree_confusion_matrix

fourfoldplot(tree_confusion_matrix$table, color = c("#8F9FCD", "#8A7FCD"))

rpart.plot(tree_best_fit$fit, extra = 101, under = TRUE, box.palette = "auto", 
           tweak = 1.20, roundint = FALSE)

tree_probabilities <- predict(tree_best_fit, test, type = "prob")
tree_curve <- roc(test$diabetes, tree_probabilities$.pred_pos)
plot.roc(tree_curve, print.auc = TRUE, auc.polygon = TRUE, asp = 0.50,
         grid = TRUE, identity.lty = 2, identity.lwd = 1, print.thres = TRUE,
         auc.polygon.col= rgb(0.1,0.1,1, alpha = 0.3), legacy.axes = TRUE,
         xlab = "FPR", ylab = "TPR", print.thres.cex = 0.7,
         print.auc.cex = 0.9)

tree_auc <- auc(tree_curve)
"AUC" -> names(tree_auc)
tree_accuracy <- tree_confusion_matrix$overall['Accuracy']
tree_sensitivity <- tree_confusion_matrix$byClass['Sensitivity']
tree_specificity <- tree_confusion_matrix$byClass['Specificity']
tree_precision <- tree_confusion_matrix$byClass[5]
tree_performance_metrics <- c(tree_auc, tree_accuracy, tree_sensitivity, 
                              tree_specificity, 
                              tree_precision)

tree_performance_metrics

tree_Var_Importance <- tree_best_fit$fit$variable.importance
tree_Var_Importance <- (tree_Var_Importance/sum(tree_Var_Importance))*100
tree_Var_Importance <- data.frame(tree_Var_Importance)
tree_Var_Importance <- data.frame(Variable =rownames(tree_Var_Importance),
                                  Importance = tree_Var_Importance$tree_Var_Importance)

ggplot(tree_Var_Importance, aes(x = Importance, y = reorder(Variable, Importance))) +
  geom_bar(stat = "identity", fill = "#8A7FCD", alpha = 0.75) +
  geom_text(aes(label = round(Importance, 2)), size = 3,
            hjust = -0.2, color = "#8A7FCD") +
  labs(x = "Importance", y = "Variable") + 
  xlim(0,55) +
  theme_minimal()


# Balanced/Oversampled approach
library(rpart)
library(rpart.plot)
library(pROC)
set.seed(123)
balanced_decision_tree <- decision_tree(
                          cost_complexity = 0.02,
                          tree_depth = tune(),
                          min_n = tune()
                                        ) %>%
                          set_mode("classification") %>%
                          set_engine(engine = "rpart", 
                                     parms = list(split = "information"))

balanced_tree_recipe <- recipe(data = balanced_train, formula = diabetes~.)

balanced_tree_workflow <- workflow() %>%
                  add_recipe(balanced_tree_recipe) %>%
                  add_model(balanced_decision_tree) 
               
balanced_tree_parameters_grid <- grid_regular(parameters(balanced_decision_tree), levels = 10)

balanced_tree_tune_process <- tune_grid(balanced_tree_workflow,
                          resamples = balanced_folds,
                          grid = balanced_tree_parameters_grid,
                          metrics = metric_set(roc_auc))

autoplot(balanced_tree_tune_process)

balanced_tree_best_process <-  select_best(balanced_tree_tune_process, metric = 'roc_auc')

balanced_best_tree <- finalize_model(balanced_decision_tree, balanced_tree_best_process)
balanced_best_tree

balanced_resamples_fit <- fit_resamples(balanced_best_tree, diabetes ~ ., 
                               resamples = balanced_folds)
collect_metrics(balanced_resamples_fit)

balanced_tree_best_fit <- fit(balanced_best_tree, diabetes~., balanced_train)
balanced_tree_best_fit

balanced_tree_pred.classes <- predict(balanced_tree_best_fit, test, type = "class")
balanced_tree_pred.classes <- factor(unlist(balanced_tree_pred.classes))
balanced_tree_confusion_matrix <- confusionMatrix(data = balanced_tree_pred.classes, 
                                         reference = test$diabetes, 
                                         positive = "pos")

balanced_tree_confusion_matrix

fourfoldplot(balanced_tree_confusion_matrix$table, color = c("#8F9FCD", "#8A7FCD"))

rpart.plot(balanced_tree_best_fit$fit, extra = 101, under = TRUE, box.palette = "auto", 
           tweak = 1.20, roundint = FALSE)

balanced_tree_probabilities <- predict(balanced_tree_best_fit, test, type = "prob")
balanced_tree_curve <- roc(test$diabetes, balanced_tree_probabilities$.pred_pos)
plot.roc(balanced_tree_curve, print.auc = TRUE, auc.polygon = TRUE, asp = 0.50,
         grid = TRUE, identity.lty = 2, identity.lwd = 1, print.thres = TRUE,
         auc.polygon.col= rgb(0.1,0.1,1, alpha = 0.3), legacy.axes = TRUE,
         xlab = "FPR", ylab = "TPR", print.thres.cex = 0.7,
         print.auc.cex = 0.9)

balanced_tree_auc <- auc(balanced_tree_curve)
"AUC" -> names(balanced_tree_auc)
balanced_tree_accuracy <- balanced_tree_confusion_matrix$overall['Accuracy']
balanced_tree_sensitivity <- balanced_tree_confusion_matrix$byClass['Sensitivity']
balanced_tree_specificity <- balanced_tree_confusion_matrix$byClass['Specificity']
balanced_tree_precision <- balanced_tree_confusion_matrix$byClass[5]
balanced_tree_performance_metrics <- c(balanced_tree_auc, balanced_tree_accuracy,   
balanced_tree_sensitivity, balanced_tree_specificity, balanced_tree_precision)

balanced_tree_performance_metrics

balanced_tree_Var_Importance <- balanced_tree_best_fit$fit$variable.importance
balanced_tree_Var_Importance <- (balanced_tree_Var_Importance /
                                 sum(balanced_tree_Var_Importance))*100
balanced_tree_Var_Importance <- data.frame(balanced_tree_Var_Importance)
balanced_tree_Var_Importance <- data.frame(
                              Variable=rownames(balanced_tree_Var_Importance),
                              Importance =     
                                balanced_tree_Var_Importance$balanced_tree_Var_Importance)

ggplot(balanced_tree_Var_Importance, aes(x = Importance, y = reorder(Variable, Importance))) +
  geom_bar(stat = "identity", fill = "#8A7FCD", alpha = 0.75) +
  geom_text(aes(label = round(Importance, 2)), size = 3,
            hjust = -0.2, color = "#8A7FCD") +
  labs(x = "Importance", y = "Variable") + 
  xlim(0,35) +
  theme_minimal()


# Bagged Decision Trees
library(baguette)
set.seed(123)
bag_tree <- bag_tree(
                     min_n = tune(),
                     tree_depth = 5,
                     cost_complexity = 0.02,     
                                        ) %>%
                      set_engine('rpart', times = 100,
                                 parms = list(split = 'information')) %>%
                      set_mode('classification')

bag_recipe <- recipe(data = train, formula = diabetes~.) 

bag_parameters_grid <- grid_regular(parameters(bag_tree), levels = 10)

bag_workflow <- workflow() %>%
                add_recipe(bag_recipe) %>%
                add_model(bag_tree)

bag_tune_process <- tune_grid(bag_workflow,
                               resamples = folds,
                               grid = bag_parameters_grid,
                               metrics = metric_set(roc_auc))

autoplot(bag_tune_process)

bag_best_process <- select_best(bag_tune_process, metric = 'roc_auc')
best_bag <- finalize_model(bag_tree, bag_best_process)
best_bag

resamples_fit <- fit_resamples(best_bag, diabetes ~ ., 
                               resamples = folds)
collect_metrics(resamples_fit)

bag_best_fit <- fit(best_bag, diabetes~., train)
bag_best_fit

bag_pred.classes <- predict(bag_best_fit, test, type = "class")
bag_pred.classes <- factor(unlist(bag_pred.classes))
bag_confusion_matrix <- confusionMatrix(data = bag_pred.classes, 
                                reference =  test$diabetes, positive = "pos")
bag_confusion_matrix

fourfoldplot(bag_confusion_matrix$table, color = c("#8F9FCD", "#8A7FCD"))

bag_probabilities <- predict(bag_best_fit, test, type = "prob")
bag_curve <- roc(test$diabetes, bag_probabilities$.pred_pos)
plot.roc(bag_curve, print.auc = TRUE, auc.polygon = TRUE, asp = 0.50,
         grid = TRUE, identity.lty = 2, identity.lwd = 1, print.thres = TRUE,
         auc.polygon.col= rgb(0.1,0.1,1, alpha = 0.3), legacy.axes = TRUE,
         xlab = "FPR", ylab = "TPR", print.thres.cex = 0.7,
         print.auc.cex = 0.9)

bag_auc <- auc(bag_curve)
"bag_AUC" -> names(bag_auc)
bag_accuracy <- bag_confusion_matrix$overall['Accuracy']
bag_sensitivity <- bag_confusion_matrix$byClass['Sensitivity']
bag_specificity <- bag_confusion_matrix$byClass['Specificity']
bag_precision <- bag_confusion_matrix$byClass[5]
bag_performance_metrics <- c(bag_auc, bag_accuracy, bag_sensitivity, 
                             bag_specificity, 
                              bag_precision)

bag_performance_metrics

bag_Var_Imp <- bag_best_fit$fit$imp[, 1:2]
bag_Var_Imp$value <- (bag_Var_Imp$value/
                             sum(bag_Var_Imp$value))*100
                                                                  
ggplot(bag_Var_Imp, aes(x = value, y =  reorder(term, value))) +
  geom_bar(stat = "identity", fill = "#8A7FCD", alpha = 0.75) + 
  geom_text(aes(label = round(value, 2)), size = 3,
            hjust = -0.2, color = "#8A7FCD") +
  labs(x = "Importance", y = "Variable") + 
  xlim(0,50) +
  theme_minimal() 


# Balanced/Oversampled approach
library(baguette)
set.seed(123)
balanced_bag_tree <- bag_tree(
                     min_n = tune(),
                     tree_depth = 5,
                     cost_complexity = 0.02,     
                                        ) %>%
                      set_engine('rpart', times = 100,
                                 parms = list(split = 'information')) %>%
                      set_mode('classification')

balanced_bag_recipe <- recipe(data = balanced_train, formula = diabetes~.) 

balanced_bag_parameters_grid <- grid_regular(parameters(balanced_bag_tree), levels = 10)

balanced_bag_workflow <- workflow() %>%
                add_recipe(balanced_bag_recipe) %>%
                add_model(balanced_bag_tree)

balanced_bag_tune_process <- tune_grid(balanced_bag_workflow,
                               resamples = balanced_folds,
                               grid = balanced_bag_parameters_grid,
                               metrics = metric_set(roc_auc))

autoplot(balanced_bag_tune_process)

balanced_bag_best_process <- select_best(balanced_bag_tune_process, metric = 'roc_auc')
balanced_best_bag <- finalize_model(balanced_bag_tree, balanced_bag_best_process)
balanced_best_bag

balanced_resamples_fit <- fit_resamples(balanced_best_bag, diabetes ~ ., 
                               resamples = balanced_folds)
collect_metrics(balanced_resamples_fit)

balanced_bag_best_fit <- fit(balanced_best_bag, diabetes~., balanced_train)
balanced_bag_best_fit

balanced_bag_pred.classes <- predict(balanced_bag_best_fit, test, type = "class")
balanced_bag_pred.classes <- factor(unlist(balanced_bag_pred.classes))
balanced_bag_confusion_matrix <- confusionMatrix(data = balanced_bag_pred.classes, 
                                reference =  test$diabetes, positive = "pos")
balanced_bag_confusion_matrix

fourfoldplot(balanced_bag_confusion_matrix$table, color = c("#8F9FCD", "#8A7FCD"))

balanced_bag_probabilities <- predict(balanced_bag_best_fit, test, type = "prob")
balanced_bag_curve <- roc(test$diabetes, balanced_bag_probabilities$.pred_pos)
plot.roc(balanced_bag_curve, print.auc = TRUE, auc.polygon = TRUE, asp = 0.50,
         grid = TRUE, identity.lty = 2, identity.lwd = 1, print.thres = TRUE,
         auc.polygon.col= rgb(0.1,0.1,1, alpha = 0.3), legacy.axes = TRUE,
         xlab = "FPR", ylab = "TPR", print.thres.cex = 0.7,
         print.auc.cex = 0.9)

balanced_bag_auc <- auc(balanced_bag_curve)
"bag_AUC" -> names(balanced_bag_auc)
balanced_bag_accuracy <- balanced_bag_confusion_matrix$overall['Accuracy']
balanced_bag_sensitivity <- balanced_bag_confusion_matrix$byClass['Sensitivity']
balanced_bag_specificity <- balanced_bag_confusion_matrix$byClass['Specificity']
balanced_bag_precision <- balanced_bag_confusion_matrix$byClass[5]
balanced_bag_performance_metrics <- c(balanced_bag_auc, balanced_bag_accuracy, balanced_bag_sensitivity, balanced_bag_specificity, balanced_bag_precision)

balanced_bag_performance_metrics

balanced_bag_Var_Imp <- balanced_bag_best_fit$fit$imp[, 1:2]
balanced_bag_Var_Imp$value <- (balanced_bag_Var_Imp$value/
                             sum(balanced_bag_Var_Imp$value))*100
                                                                  
ggplot(balanced_bag_Var_Imp, aes(x = value, y =  reorder(term, value))) +
  geom_bar(stat = "identity", fill = "#8A7FCD", alpha = 0.75) + 
  geom_text(aes(label = round(value, 2)), size = 3,
            hjust = -0.2, color = "#8A7FCD") +
  labs(x = "Importance", y = "Variable") + 
  xlim(0,30) + 
  theme_minimal() 


# Random Forest
# Original/Unbalanced approach
library(ranger)
set.seed(123)
forest <- rand_forest(
                      mtry = 3,
                      trees = 500, 
                      min_n = tune()
                                      ) %>%
                      set_engine("ranger", num.threads = 8,
                       splitrule = "gini", importance = "impurity_corrected") %>%
                      set_mode('classification')

forest_recipe <- recipe(data = train, formula = diabetes~.)
  
forest_parameters <- extract_parameter_set_dials(forest) %>%
                      finalize(train)

forest_grid <- grid_regular(forest_parameters$object, levels = 10)

forest_workflow <- workflow() %>%
                    add_recipe(forest_recipe) %>%
                    add_model(forest)

tune_forest <- tune_grid(forest_workflow,
                         resamples = folds,
                         grid = forest_grid,
                         metrics = metric_set(roc_auc))

autoplot(tune_forest)

forest_best_process <- select_best(tune_forest, metric = 'roc_auc')
best_forest <- finalize_model(forest, forest_best_process)
best_forest

resamples_fit <- fit_resamples(best_forest, diabetes ~ ., 
                               resamples = folds)
collect_metrics(resamples_fit)

forest_best_fit <- fit(best_forest, diabetes~., train)
forest_best_fit

forest_pred.classes <- predict(forest_best_fit, test, type = "class")
forest_pred.classes <- factor(unlist(forest_pred.classes))
forest_confusion_matrix <- confusionMatrix(forest_pred.classes, test$diabetes, 
                                        positive = "pos")
forest_confusion_matrix

fourfoldplot(forest_confusion_matrix$table, color = c("#8F9FCD", "#8A7FCD"))

forest_probabilities <- predict(forest_best_fit, test, type = "prob")
forest_curve <- roc(test$diabetes, forest_probabilities$.pred_pos)
plot.roc(forest_curve, print.auc = TRUE, auc.polygon = TRUE, asp = 0.50,
         grid = TRUE, identity.lty = 2, identity.lwd = 1, print.thres = TRUE,
         auc.polygon.col= rgb(0.1,0.1,1, alpha = 0.3), legacy.axes = TRUE,
         xlab = "FPR", ylab = "TPR", print.thres.cex = 0.7,
         print.auc.cex = 0.9)

forest_auc <- auc(forest_curve)
"forest_AUC" -> names(forest_auc)
forest_accuracy <- forest_confusion_matrix$overall['Accuracy']
forest_sensitivity <- forest_confusion_matrix$byClass['Sensitivity']
forest_specificity <- forest_confusion_matrix$byClass['Specificity']
forest_precision <- forest_confusion_matrix$byClass[5]
forest_performance_metrics <- c(forest_auc, forest_accuracy, 
                                forest_sensitivity, 
                                forest_specificity, forest_precision)

forest_performance_metrics

forest_Var_Imp <- forest_best_fit$fit$variable.importance
forest_Var_Imp <- data.frame(forest_Var_Imp)
term <- rownames(forest_Var_Imp) 
value <- forest_Var_Imp$forest_Var_Imp
forest_Var_Imp <- data.frame(term, value)

forest_Var_Imp$value <- (forest_Var_Imp$value/sum(forest_Var_Imp$value))*100

ggplot(forest_Var_Imp, aes(x = value, y =  reorder(term, value))) +
  geom_bar(stat = "identity", fill = "#8A7FCD", alpha = 0.75) + 
  geom_text(aes(label = round(value, 2)), size = 3,
            hjust = -0.2, color = "#8A7FCD") +
  labs(x = "Importance", y = "Variable") +
  xlim(0,50) + 
   theme_minimal()


# Balanced/Oversampled approach
library(ranger)
set.seed(123)
balanced_forest <- rand_forest(
                      mtry = 3,
                      trees = 500, 
                      min_n = tune()
                                      ) %>%
                      set_engine("ranger", num.threads = 8,
                       splitrule = "gini", importance = "impurity_corrected") %>%
                      set_mode('classification')

balanced_forest_recipe <- recipe(data = balanced_train, formula = diabetes~.)
  
balanced_forest_parameters <- extract_parameter_set_dials(balanced_forest) %>%
                      finalize(balanced_train)

balanced_forest_grid <- grid_regular(balanced_forest_parameters$object, levels = 10)

balanced_forest_workflow <- workflow() %>%
                    add_recipe(balanced_forest_recipe) %>%
                    add_model(balanced_forest)

balanced_tune_forest <- tune_grid(balanced_forest_workflow,
                         resamples = balanced_folds,
                         grid = balanced_forest_grid,
                         metrics = metric_set(roc_auc))

autoplot(balanced_tune_forest)

balanced_forest_best_process <- select_best(balanced_tune_forest, metric = 'roc_auc')
balanced_best_forest <- finalize_model(balanced_forest, balanced_forest_best_process)
balanced_best_forest

balanced_resamples_fit <- fit_resamples(balanced_best_forest, diabetes ~ ., 
                               resamples = balanced_folds)
collect_metrics(balanced_resamples_fit)

balanced_forest_best_fit <- fit(balanced_best_forest, diabetes~., balanced_train)
balanced_forest_best_fit

balanced_forest_pred.classes <- predict(balanced_forest_best_fit, test, type = "class")
balanced_forest_pred.classes <- factor(unlist(balanced_forest_pred.classes))
balanced_forest_confusion_matrix <- confusionMatrix(balanced_forest_pred.classes, test$diabetes, 
                                        positive = "pos")
balanced_forest_confusion_matrix

fourfoldplot(balanced_forest_confusion_matrix$table, color = c("#8F9FCD", "#8A7FCD"))

balanced_forest_probabilities <- predict(balanced_forest_best_fit, test, type = "prob")
balanced_forest_curve <- roc(test$diabetes, balanced_forest_probabilities$.pred_pos)
plot.roc(balanced_forest_curve, print.auc = TRUE, auc.polygon = TRUE, asp = 0.50,
         grid = TRUE, identity.lty = 2, identity.lwd = 1, print.thres = TRUE,
         auc.polygon.col= rgb(0.1,0.1,1, alpha = 0.3), legacy.axes = TRUE,
         xlab = "FPR", ylab = "TPR", print.thres.cex = 0.7,
         print.auc.cex = 0.9)

balanced_forest_auc <- auc(balanced_forest_curve)
"forest_AUC" -> names(balanced_forest_auc)
balanced_forest_accuracy <- balanced_forest_confusion_matrix$overall['Accuracy']
balanced_forest_sensitivity <- balanced_forest_confusion_matrix$byClass['Sensitivity']
balanced_forest_specificity <- balanced_forest_confusion_matrix$byClass['Specificity']
balanced_forest_precision <- balanced_forest_confusion_matrix$byClass[5]
balanced_forest_performance_metrics <- c(balanced_forest_auc, balanced_forest_accuracy, 
                                balanced_forest_sensitivity, 
                                balanced_forest_specificity, balanced_forest_precision)

balanced_forest_performance_metrics

balanced_forest_Var_Imp <- balanced_forest_best_fit$fit$variable.importance
balanced_forest_Var_Imp <- data.frame(balanced_forest_Var_Imp)
term <- rownames(balanced_forest_Var_Imp) 
value <- balanced_forest_Var_Imp$balanced_forest_Var_Imp
balanced_forest_Var_Imp <- data.frame(term, value)

balanced_forest_Var_Imp$value <- (balanced_forest_Var_Imp$value/sum(balanced_forest_Var_Imp$value))*100

ggplot(balanced_forest_Var_Imp, aes(x = value, y =  reorder(term, value))) +
  geom_bar(stat = "identity", fill = "#8A7FCD", alpha = 0.75) + 
  geom_text(aes(label = round(value, 2)), size = 3,
            hjust = -0.2, color = "#8A7FCD") +
  labs(x = "Importance", y = "Variable") +
  xlim(0,30) + 
   theme_minimal()


# Gradient Boosted Trees
# Original/Unbalanced approach
  library(xgboost)
set.seed(123)
boost_spec <- boost_tree(
  trees = 500,
  tree_depth = tune(), 
  min_n = tune(), 
  mtry = 3,
  loss_reduction = 2.25,  
  learn_rate = 0.01,
  sample_size = 0.80,
  stop_iter = 3) %>%
  set_engine('xgboost', nthread = 8, importance = TRUE) %>%
  set_mode('classification')

boost_parameters <- extract_parameter_set_dials(boost_spec) %>%
  finalize(train) 

boost_grid <- grid_regular(boost_parameters$object, levels = 10)

boost_recipe <- recipe(formula = diabetes ~ ., data = train) 

boost_workflow <- workflow() %>% 
  add_recipe(boost_recipe) %>% 
  add_model(boost_spec)  

tune_boost <- tune_grid(boost_workflow,
                        resamples = folds,
                        grid = boost_grid,
                        metrics = metric_set(roc_auc))

autoplot(tune_boost)

boost_best_process <- select_best(tune_boost, metric = 'roc_auc')

best_boost <- finalize_model(boost_spec, boost_best_process)
best_boost

resamples_fit <- fit_resamples(best_boost, diabetes ~ ., 
                               resamples = folds)
collect_metrics(resamples_fit)

boost_best_fit <- fit(best_boost, diabetes~., train)
boost_pred.classes <- predict(boost_best_fit, test, type = "class")
boost_pred.classes <- factor(unlist(boost_pred.classes))
boost_confusion_matrix <- confusionMatrix(data = boost_pred.classes, 
                                          reference = test$diabetes, positive = "pos")
boost_confusion_matrix

fourfoldplot(boost_confusion_matrix$table, color = c("#8F9FCD", "#8A7FCD"))

boost_probabilities <- predict(boost_best_fit, test, type = "prob")
boost_curve <- roc(test$diabetes, boost_probabilities$.pred_pos)
plot.roc(boost_curve, print.auc = TRUE, auc.polygon = TRUE, asp = 0.50,
         grid = TRUE, identity.lty = 2, identity.lwd = 1, print.thres = TRUE,
         auc.polygon.col= rgb(0.1,0.1,1, alpha = 0.3), legacy.axes = TRUE,
         xlab = "FPR", ylab = "TPR", print.thres.cex = 0.7,
         print.auc.cex = 0.9)

boost_auc <- auc(boost_curve)
"boost_AUC" -> names(boost_auc)
boost_accuracy <- boost_confusion_matrix$overall['Accuracy']
boost_sensitivity <- boost_confusion_matrix$byClass['Sensitivity']
boost_specificity <- boost_confusion_matrix$byClass['Specificity']
boost_precision <- boost_confusion_matrix$byClass[5]
boost_performance_metrics <- c(boost_auc, boost_accuracy, 
                               boost_sensitivity, 
                               boost_specificity, boost_precision)

boost_performance_metrics

boost_importance <- xgb.importance(boost_best_fit$fit$feature_names, boost_best_fit$fit)

boost_Var_Imp <- data.frame(boost_importance$Feature, boost_importance$Gain)

ggplot(boost_Var_Imp, aes(x = boost_importance$Gain*100, 
                          y =  reorder(boost_importance$Feature,
                                       boost_importance$Gain))) +
  geom_bar(stat = "identity", fill = "#8A7FCD", alpha = 0.75) + 
  geom_text(aes(label = round(boost_importance$Gain*100, 2)), size = 3,
            hjust = -0.2, color = "#8A7FCD") +
  labs(x = "Importance", y = "Variable") + 
  xlim(0,55) +
  theme_minimal() 

# Balanced/Oversampled approach
library(xgboost)
set.seed(123)
balanced_boost_spec <- boost_tree(
  trees = 500,
  tree_depth = tune(), 
  min_n = tune(), 
  mtry = 3,
  loss_reduction = 2.25,  
  learn_rate = 0.01,
  sample_size = 0.80,
  stop_iter = 3) %>%
  set_engine('xgboost', nthread = 8, importance = TRUE) %>%
  set_mode('classification')

balanced_boost_parameters <- extract_parameter_set_dials(balanced_boost_spec) %>%
  finalize(balanced_train) 

balanced_boost_grid <- grid_regular(balanced_boost_parameters$object, levels = 10)

balanced_boost_recipe <- recipe(formula = diabetes ~ ., data = balanced_train) 

balanced_boost_workflow <- workflow() %>% 
  add_recipe(balanced_boost_recipe) %>% 
  add_model(balanced_boost_spec)  

balanced_tune_boost <- tune_grid(balanced_boost_workflow,
                                 resamples = balanced_folds,
                                 grid = balanced_boost_grid,
                                 metrics = metric_set(roc_auc))

autoplot(balanced_tune_boost)

balanced_boost_best_process <- select_best(balanced_tune_boost, metric = 'roc_auc')

balanced_best_boost <- finalize_model(balanced_boost_spec, balanced_boost_best_process)
best_boost

balanced_resamples_fit <- fit_resamples(balanced_best_boost, diabetes ~ ., 
                                        resamples = balanced_folds)
collect_metrics(balanced_resamples_fit)

balanced_boost_best_fit <- fit(balanced_best_boost, diabetes~., balanced_train)
balanced_boost_pred.classes <- predict(balanced_boost_best_fit, test, type = "class")
balanced_boost_pred.classes <- factor(unlist(balanced_boost_pred.classes))
balanced_boost_confusion_matrix <- confusionMatrix(data = balanced_boost_pred.classes, 
                                                   reference = test$diabetes, positive = "pos")
balanced_boost_confusion_matrix

fourfoldplot(balanced_boost_confusion_matrix$table, color = c("#8F9FCD", "#8A7FCD"))

balanced_boost_probabilities <- predict(balanced_boost_best_fit, test, type = "prob")
balanced_boost_curve <- roc(test$diabetes, balanced_boost_probabilities$.pred_pos)
plot.roc(balanced_boost_curve, print.auc = TRUE, auc.polygon = TRUE, asp = 0.50,
         grid = TRUE, identity.lty = 2, identity.lwd = 1, print.thres = TRUE,
         auc.polygon.col= rgb(0.1,0.1,1, alpha = 0.3), legacy.axes = TRUE,
         xlab = "FPR", ylab = "TPR", print.thres.cex = 0.7,
         print.auc.cex = 0.9)

balanced_boost_auc <- auc(balanced_boost_curve)
"boost_AUC" -> names(balanced_boost_auc)
balanced_boost_accuracy <- balanced_boost_confusion_matrix$overall['Accuracy']
balanced_boost_sensitivity <- balanced_boost_confusion_matrix$byClass['Sensitivity']
balanced_boost_specificity <- balanced_boost_confusion_matrix$byClass['Specificity']
balanced_boost_precision <- balanced_boost_confusion_matrix$byClass[5]
balanced_boost_performance_metrics <- c(balanced_boost_auc, balanced_boost_accuracy, 
                                        balanced_boost_sensitivity, 
                                        balanced_boost_specificity, boost_precision)

balanced_boost_performance_metrics

balanced_boost_importance <- xgb.importance(balanced_boost_best_fit$fit$feature_names, balanced_boost_best_fit$fit)

balanced_boost_Var_Imp <- data.frame(balanced_boost_importance$Feature, balanced_boost_importance$Gain)

ggplot(balanced_boost_Var_Imp, aes(x = balanced_boost_importance$Gain*100, 
                                   y =  reorder(balanced_boost_importance$Feature,
                                                balanced_boost_importance$Gain))) +
  geom_bar(stat = "identity", fill = "#8A7FCD", alpha = 0.75) + 
  geom_text(aes(label = round(balanced_boost_importance$Gain*100, 2)), size = 3,
            hjust = -0.2, color = "#8A7FCD") +
  labs(x = "Importance", y = "Variable") + 
  xlim(0,25) + 
  theme_minimal() 

# FINAL TABLE PERFORMANCE
table_performance <- data.frame(tree = round(tree_performance_metrics, 2),
                                balanced_tree = round(balanced_tree_performance_metrics, 2),
                                bagging = round(bag_performance_metrics, 2),
                                balanced_bagging = round(balanced_bag_performance_metrics, 2),
                                forest = round(forest_performance_metrics, 2),  
                                balanced_forest = round(balanced_forest_performance_metrics, 2),
                                boosted = round(boost_performance_metrics, 2),
                                balanced_boosted = round(balanced_boost_performance_metrics, 2))

table_performance




