# Tuning the loss function along with the network
#
# Survey data are mostly zeros: most surveys catch nothing, and the few that
# do catch something vary a lot in how much. Two loss functions suit that
# kind of response, and each has settings that are usually left at their
# defaults. This script treats those settings as hyperparameters and lets
# Keras Tuner search them together with the network.
#   1. Any catch? A classifier trained with binary focal cross-entropy,
#      tuning its class weight (alpha) and focusing parameter (gamma).
#   2. How much? A regression network trained with a Tweedie loss, tuning
#      its power between 1 and 2.
# The surveys are simulated from a Tweedie distribution with a known power,
# so nothing here depends on real data.
# Run from the repository root: Rscript tuned-loss-functions/tune_loss_functions.R
# Needs the keras3 package. The first run downloads TensorFlow and Keras
# Tuner into a private Python environment, which takes a few minutes.

# Packages ------

# Load the R interface to Keras
suppressPackageStartupMessages(library(keras3))

# Ask for Keras Tuner, and the two packages its Bayesian search and logging use,
# in the Python environment that keras3 manages
reticulate::py_require(c("keras-tuner", "scikit-learn", "tensorboard"))

# Import Keras Tuner, which has no R wrapper of its own here
kt <- reticulate::import("keras_tuner")

# Settings ------

# Set how many surveys to simulate and how they are split
n_surveys <- 4000
n_train <- 2400
n_val <- 800

# Set the true Tweedie power and dispersion of the simulated catch
true_power <- 1.3
dispersion <- 2

# Set the size of each search; the paper this follows ran 1,000 trials per network
n_trials <- 12
n_epochs <- 40

# Fix the seeds for the simulation and for the networks
set.seed(2026)
set_random_seed(2026)

# Simulated surveys ------

# Draw four standardized survey covariates
covariates <- matrix(rnorm(n_surveys * 4), ncol = 4,
                     dimnames = list(NULL, c("temperature", "water_level", "humidity", "effort")))

# Give each survey an expected catch that depends on the covariates
expected_catch <- exp(-0.6 + 0.7 * covariates[, "temperature"] - 0.4 * covariates[, "water_level"]^2 +
                        0.4 * covariates[, "humidity"] * covariates[, "effort"] + 0.6 * covariates[, "effort"])

# Convert the mean, power and dispersion to the Poisson and gamma pieces of a Tweedie draw
catch_rate <- expected_catch^(2 - true_power) / (dispersion * (2 - true_power))
catch_shape <- (2 - true_power) / (true_power - 1)
catch_scale <- dispersion * (true_power - 1) * expected_catch^(true_power - 1)

# Draw the number of catches per survey, then the total caught as a sum of gamma amounts
n_catches <- rpois(n_surveys, catch_rate)
catch <- rgamma(n_surveys, shape = catch_shape * n_catches, scale = catch_scale)

# Record whether each survey caught anything
any_catch <- as.numeric(catch > 0)

# Report how zero-heavy the simulated surveys are
cat(sprintf("Surveys with no catch: %.0f%%. Average catch when there is one: %.2f\n",
            100 * mean(any_catch == 0), mean(catch[catch > 0])))

# Split the surveys into training, validation and test sets
train_rows <- 1:n_train
val_rows <- (n_train + 1):(n_train + n_val)
test_rows <- (n_train + n_val + 1):n_surveys

# Loss function ------

# Tweedie loss with the power left open, so the tuner can choose it
tweedie_loss <- function(power) {

  # Return a loss with this power fixed inside it
  function(y_true, y_pred) {

    # Keep predictions above zero so the powers below are defined
    y_pred <- op_maximum(y_pred, config_epsilon())

    # Tweedie deviance up to terms that do not depend on the prediction
    -y_true * op_power(y_pred, 1 - power) / (1 - power) + op_power(y_pred, 2 - power) / (2 - power)
  }
}

# Hypermodels ------

# Build a small network with the layer width and learning rate left to the tuner
build_network <- function(hp, output_activation) {

  # Let the tuner choose the width of the two hidden layers
  units <- hp$Choice("units", c(16L, 32L, 64L))

  # Stack two hidden layers and one output
  keras_model_sequential(input_shape = ncol(covariates)) %>%
    layer_dense(units = units, activation = "relu") %>%
    layer_dense(units = units, activation = "relu") %>%
    layer_dense(units = 1, activation = output_activation)
}

# Classifier for any catch, with the focal loss settings as hyperparameters
build_catch_model <- function(hp) {

  # Let the tuner choose the class weight and the focusing parameter of the focal loss
  alpha <- hp$Float("alpha", min_value = 0.05, max_value = 0.95, step = 0.05)
  gamma <- hp$Float("gamma", min_value = 0, max_value = 5, step = 0.5)

  # Build the network with a probability output
  model <- build_network(hp, output_activation = "sigmoid")

  # Compile with the tuned loss and a metric that does not depend on the loss
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = hp$Choice("learning_rate", c(0.01, 0.001))),
    loss = loss_binary_focal_crossentropy(apply_class_balancing = TRUE, alpha = alpha, gamma = gamma),
    metrics = list(metric_auc(name = "auc"))
  )

  # Return the compiled model to the tuner
  model
}

# Regression network for the amount caught, with the Tweedie power as a hyperparameter
build_amount_model <- function(hp) {

  # Let the tuner choose the Tweedie power
  power <- hp$Float("power", min_value = 1.01, max_value = 1.99, step = 0.01)

  # Build the network with an output that is always positive
  model <- build_network(hp, output_activation = "softplus")

  # Compile with the tuned loss and a metric that does not depend on the loss
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = hp$Choice("learning_rate", c(0.01, 0.001))),
    loss = tweedie_loss(power),
    metrics = list(metric_root_mean_squared_error(name = "rmse"))
  )

  # Return the compiled model to the tuner
  model
}

# Search ------

# Run one Bayesian search and return the tuner
run_search <- function(build_model, response, objective_name, direction) {

  # Score trials on a validation metric, never on the loss: losses with
  # different settings are on different scales and cannot be compared
  tuner <- kt$BayesianOptimization(
    hypermodel = build_model,
    objective = kt$Objective(objective_name, direction = direction),
    max_trials = as.integer(n_trials),
    seed = 2026L,
    directory = tempdir(),
    project_name = objective_name,
    overwrite = TRUE
  )

  # Train every trial the same way, stopping early when validation stops improving.
  # The response goes in as a one-column matrix so Python receives an array, not a list
  tuner$search(
    x = covariates[train_rows, ],
    y = as.matrix(response[train_rows]),
    validation_data = reticulate::tuple(covariates[val_rows, ], as.matrix(response[val_rows])),
    epochs = as.integer(n_epochs),
    batch_size = 64L,
    callbacks = list(callback_early_stopping(monitor = objective_name, mode = direction,
                                             patience = 5, restore_best_weights = TRUE)),
    verbose = 0L
  )

  # Return the finished tuner
  tuner
}

# List the best trials of a search as a table
best_trials <- function(tuner, n = 5) {

  # Take the best trials in order
  trials <- tuner$oracle$get_best_trials(as.integer(n))

  # Hold one row per trial
  rows <- vector("list", length(trials))

  # Record the settings and validation score of each trial
  for (i in seq_along(trials)) {
    rows[[i]] <- data.frame(trials[[i]]$hyperparameters$values, validation_score = round(trials[[i]]$score, 4))
  }

  # Stack the trials into one table
  do.call(rbind, rows)
}

# Search the classifier, scored on validation AUC
catch_tuner <- run_search(build_catch_model, any_catch, objective_name = "val_auc", direction = "max")

# Search the regression network, scored on validation root mean squared error
amount_tuner <- run_search(build_amount_model, catch, objective_name = "val_rmse", direction = "min")

# Results ------

# Show the best settings found for the focal loss
cat("\nAny catch? Best trials by validation AUC\n")
print(best_trials(catch_tuner), row.names = FALSE)

# Show the best settings found for the Tweedie loss
cat(sprintf("\nHow much? Best trials by validation RMSE (the catch was simulated at power %.1f)\n", true_power))
print(best_trials(amount_tuner), row.names = FALSE)

# Take the best network from each search; reloading one warns that the optimizer state is not restored, which is fine here
catch_model <- catch_tuner$get_best_models(1L)[[1]]
amount_model <- amount_tuner$get_best_models(1L)[[1]]

# Score both on the test surveys, which the search never saw
catch_test <- evaluate(catch_model, covariates[test_rows, ], any_catch[test_rows], verbose = 0)
amount_test <- evaluate(amount_model, covariates[test_rows, ], catch[test_rows], verbose = 0)

# Compare the regression network with predicting the training mean for every survey
baseline_rmse <- sqrt(mean((catch[test_rows] - mean(catch[train_rows]))^2))

# Report the held-out scores
cat(sprintf("\nTest AUC of the tuned classifier: %.3f\n", catch_test$auc))
cat(sprintf("Test RMSE of the tuned regression network: %.3f (predicting the mean: %.3f)\n",
            amount_test$rmse, baseline_rmse))
