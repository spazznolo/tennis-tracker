
library(tidyverse)

houghs <- 
  read_csv('~/Documents/tennis-tracker/assets/temp/hough_lines.csv') %>%
  mutate(
    x_min = ifelse(x1 < x2, x1, x2),
    x_max = ifelse(x1 > x2, x1, x2),
    y_min = ifelse(y1 < y2, y1, y2),
    y_max = ifelse(y1 > y2, y1, y2),
    slope = (y2 - y1)/(x2 -x1),
    length = sqrt((x_max - x_min)^2 + (y_max - y_min)^2)
    )

length_lines <-
  houghs %>% 
  filter(abs(slope) > 1.5, abs(slope) < 3, length > 100)

hist(length_lines$length)

width_lines <-
  houghs %>%
  filter(
    abs(slope) < 0.15 & length > 25 & x_min > 180 & x_max < 800 &
      ((y_max < max(length_lines$y_max) + 10 & y_max > max(length_lines$y_max) - 10) |
         (y_min < min(length_lines$y_min) + 5 & y_min > min(length_lines$y_min) - 5)))

width_lines %>%
  ggplot() +
  geom_histogram(aes(y1))
  
length_lines %>%
  ggplot() +
  geom_histogram(aes(x2))
