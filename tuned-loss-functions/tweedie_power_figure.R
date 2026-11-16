# What the Tweedie power does to python survey catches
#
# A Tweedie response answers two questions about a survey in one
# distribution: whether it caught anything, and how much it caught when it
# did. The power sets the balance between the two at the same average.
# Top panel: the distribution of catch among surveys with a catch at three
# powers. Bottom panels: the share of surveys with no catch and the average
# catch when there is one, across the whole range of the power, with the
# three powers marked in their colours.
# Everything is computed from the Tweedie distribution itself (compound
# Poisson-gamma). Nothing is fitted and no survey data are used.
# Run from the repository root: Rscript tuned-loss-functions/tweedie_power_figure.R

# Packages ------

# Load the packages used for data manipulation, plotting and panel layout
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
})

# Settings ------

# Fix the shared mean and dispersion so only the power changes
target_mean <- 1.5
dispersion <- 4

# Choose three Tweedie powers between 1 and 2
powers <- c(1.2, 1.5, 1.8)
power_names <- paste("Power", powers)

# Give the three powers three shades of blue, light to dark
power_colours <- c("#74A9CF", "#2B8CBE", "#045A8D")
names(power_colours) <- power_names

# Set the colours of the curves across the power and of the point outlines, and the font
curve_colour <- "#74A9CF"
outline_colour <- "#023858"
font <- "Arial"

# Set the catch values the top panel is evaluated at, starting just above zero
catch_grid <- seq(0.01, 13, by = 0.02)

# Set the top of the top panel; the highest power is unbounded near zero and leaves it
density_top <- 0.42

# Set the output file beside this script
out_file <- file.path("tuned-loss-functions", "tweedie_power_figure.png")

# Tweedie pieces ------

# Poisson rate of catches per survey at a given power
catch_rate <- function(p) {
  target_mean^(2 - p) / (dispersion * (2 - p))
}

# Gamma shape of one catch at a given power
catch_shape <- function(p) {
  (2 - p) / (p - 1)
}

# Gamma scale of one catch at a given power
catch_scale <- function(p) {
  dispersion * (p - 1) * target_mean^(p - 1)
}

# Share of surveys with no catch at a given power: the chance of zero catches
no_catch_share <- function(p) {
  exp(-catch_rate(p))
}

# Density of the survey total among surveys with a catch, at chosen totals
catch_density <- function(totals, p) {

  # Weight each possible number of catches by its chance, given at least one
  n_catches <- 1:80
  weights <- dpois(n_catches, catch_rate(p)) / (1 - no_catch_share(p))

  # Start from zero density at every total
  density <- numeric(length(totals))

  # Add the gamma density of the total for each number of catches
  for (i in seq_along(n_catches)) {
    density <- density + weights[i] * dgamma(totals, shape = catch_shape(p) * n_catches[i],
                                             scale = catch_scale(p))
  }

  # Return the density at each total
  density
}

# Curves ------

# Hold the catch distribution of each power
density_list <- vector("list", length(powers))

# Compute the distribution of catch among surveys with a catch at each power
for (i in seq_along(powers)) {
  density_list[[i]] <- data.frame(power = power_names[i], catch = catch_grid,
                                  density = catch_density(catch_grid, powers[i]))
}

# Stack the distributions and keep the powers in order
density_curves <- bind_rows(density_list) %>%
  mutate(power = factor(power, levels = power_names))

# Evaluate both parts of the response across the range of the power
power_curve <- data.frame(power_value = seq(1.01, 1.99, by = 0.01)) %>%
  mutate(no_catch = no_catch_share(power_value),
         mean_when_caught = target_mean / (1 - no_catch))

# Evaluate the same two parts at the three powers drawn as distributions
power_points <- data.frame(power_value = powers,
                           power = factor(power_names, levels = power_names)) %>%
  mutate(no_catch = no_catch_share(power_value),
         mean_when_caught = target_mean / (1 - no_catch))

# Checks ------

# Report the two parts at the three powers
print(power_points %>%
        mutate(across(where(is.numeric), ~ round(.x, 3))))

# Confirm each distribution integrates to one and reproduces the average catch when there is one
for (i in seq_along(powers)) {
  total_area <- integrate(catch_density, lower = 0, upper = Inf, p = powers[i])$value
  mean_catch <- integrate(function(x) x * catch_density(x, powers[i]), lower = 0, upper = Inf)$value
  print(round(c(power = powers[i], area = total_area, mean_when_caught = mean_catch), 3))
}

# Plot ------

# Style every panel the same way: a plain boxed panel with black text
theme_panels <- theme_classic(base_size = 14) +
  theme(text = element_text(family = font),
        axis.text = element_text(colour = "black", size = 12.5),
        axis.title = element_text(colour = "black", size = 14.5),
        axis.title.x = element_text(margin = margin(t = 8.4)),
        axis.title.y = element_text(margin = margin(r = 8.4)),
        axis.ticks = element_line(colour = "black"),
        axis.line = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.7),
        plot.title = element_text(size = 14.5, face = "bold", hjust = 0.5,
                                  margin = margin(b = 6)),
        plot.margin = margin(8, 16, 8, 10))

# Place a direct label beside each distribution, clear of the other curves
curve_labels <- data.frame(power = factor(power_names, levels = power_names),
                           catch = c(4.9, 2.2, 0.7), density = c(0.2, 0.3, 0.38))

# Top panel: overlapping distributions of catch among surveys with a catch
panel_catch <- ggplot(density_curves, aes(x = catch, y = pmin(density, density_top * 1.2))) +
  geom_area(aes(fill = power), alpha = 0.35, position = "identity") +
  geom_line(aes(colour = power), linewidth = 0.9) +
  geom_text(data = curve_labels, aes(y = density, label = power, colour = power), hjust = 0,
            size = 4.6, family = font, fontface = "bold") +
  scale_fill_manual(values = power_colours, guide = "none") +
  scale_colour_manual(values = power_colours, guide = "none") +
  scale_x_continuous(breaks = seq(0, 12, by = 2)) +
  coord_cartesian(xlim = c(0, 12.5), ylim = c(0, density_top), expand = FALSE) +
  labs(x = "Python catch per survey", y = "Share of surveys\nwith a catch",
       title = "Python surveys with a catch") +
  theme_panels +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

# Bottom left: share of surveys with no catch across the power, three powers marked
panel_no_catch <- ggplot(power_curve, aes(x = power_value, y = no_catch)) +
  geom_line(colour = curve_colour, linewidth = 1.2) +
  geom_point(data = power_points, aes(fill = power), shape = 21, colour = outline_colour,
             stroke = 0.7, size = 4.2) +
  scale_fill_manual(values = power_colours, guide = "none") +
  scale_x_continuous(breaks = seq(1, 2, by = 0.2)) +
  scale_y_continuous(labels = scales::label_percent(), breaks = seq(0, 1, by = 0.25)) +
  coord_cartesian(xlim = c(1, 2), ylim = c(0, 1), expand = FALSE) +
  labs(x = "Tweedie power", y = "Surveys with\nno catch", title = "Any catch?") +
  theme_panels

# Bottom right: average catch among surveys with a catch across the power
panel_mean <- ggplot(power_curve, aes(x = power_value, y = mean_when_caught)) +
  geom_line(colour = curve_colour, linewidth = 1.2) +
  geom_point(data = power_points, aes(fill = power), shape = 21, colour = outline_colour,
             stroke = 0.7, size = 4.2) +
  scale_fill_manual(values = power_colours, guide = "none") +
  scale_x_continuous(breaks = seq(1, 2, by = 0.2)) +
  scale_y_continuous(breaks = seq(0, 6, by = 2)) +
  coord_cartesian(xlim = c(1, 2), ylim = c(0, 6), expand = FALSE) +
  labs(x = "Tweedie power", y = "Average catch\nwhen there is one", title = "How many?") +
  theme_panels

# Put the distributions over the two parts they split into
fig <- panel_catch / (panel_no_catch + panel_mean) +
  plot_layout(heights = c(1, 0.85))

# Save ------

# Export at 1200 x 1200 px with ragg for clean type
ggsave(filename = out_file, plot = fig, width = 6, height = 6, units = "in", dpi = 200,
       device = ragg::agg_png)
