
library(tidyverse)

houghs <- 
  read_csv('~/Documents/tennis-tracker/hough_lines.csv') %>%
  mutate(
    slope_y = (y2 - y1)/(x2 -x1),
    length = sqrt((x1 - x2)^2 + (y1 - y2)^2)
    )

length_lines <-
  houghs %>% 
  filter(abs(slope_y) > 1, abs(slope_y) < 5)

houghs %>%
  filter(abs(slope_y) < 1, x1 < 1000, x1 > 500)
  
length_lines %>%
  ggplot() +
  geom_histogram(aes(x2))
