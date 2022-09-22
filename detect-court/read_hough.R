
library(tidyverse)

houghs <- 
  read_csv('~/Documents/tennis-tracker/assets/temp/hough_lines.csv') %>%
  mutate(
    x_min = ifelse(x1 < x2, x1, x2),
    x_max = ifelse(x1 > x2, x1, x2),
    y_min = ifelse(y1 < y2, y1, y2),
    y_max = ifelse(y1 > y2, y1, y2),
    slope_y = (y_max - y_min)/(x_max -x_min),
    length = sqrt((x_max - x_min)^2 + (y_max - y_min)^2)
    ) %>%
  select(x_min:length)

length_lines <-
  houghs %>% 
  filter(abs(slope_y) > 1.5, abs(slope_y) < 3, length > 250)

hist(length_lines$length)

width_lines <-
  houghs %>%
  filter(
    abs(slope_y) < 0.1 &
      ((y_max < max(length_lines$y_max) + 2 & y_max > max(length_lines$y_max) - 2) |
         (y_min < min(length_lines$y_min) + 2 & y_min > min(length_lines$y_min) - 2)))

width_lines %>%
  ggplot() +
  geom_histogram(aes(y1))
  
length_lines %>%
  ggplot() +
  geom_histogram(aes(x2))
