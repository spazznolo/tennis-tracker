
library(tidyverse)
library(forecast)

close_locations_cleaned <- 
  read_csv('~/Documents/tennis-tracker/assets/demo/close_location.csv') %>%
  filter(chunk != 0) %>%
  mutate(across(frame:y, ~ifelse(. == 0, NA, .)), pres = 1) %>%
  arrange(chunk, frame) %>%
  group_by(chunk) %>%
  mutate(dist = sqrt(((x - lag(x))^2) + ((y - lag(y))^2))) %>%
  mutate(x = ifelse(dist > 6, NA, x), y = ifelse(dist > 6, NA, y)) %>%
  mutate(
    x = as.vector(round(na.interp(ts(x, frequency = 1)), 0)),
    y = as.vector(round(na.interp(ts(y, frequency = 1)), 0))
  ) %>%
  ungroup() %>%
  select(chunk, frame, x, y)

write_csv(close_locations_cleaned, '~/Documents/tennis-tracker/assets/demo/close_location_clean.csv')

