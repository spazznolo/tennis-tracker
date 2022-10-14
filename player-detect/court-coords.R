
library(tidyverse)
library(forecast)

court_coords_cleaned <- 
  read_csv('~/Documents/tennis-tracker/assets/demo/court_coords.csv') %>%
  filter(frame > 0) %>%
  mutate(across(frame:y_4, ~ifelse(. == 0, NA, .))) %>%
  arrange(chunk, frame) %>%
  group_by(chunk) %>%
  mutate(
    across(x_1:y_4, ~ifelse(abs(. - lag(.)) > 3, NA, .)),
    across(x_1:y_4, ~as.vector(round(na.interp(ts(., frequency = 1)), 0)))
    ) %>%
  ungroup() #%>%
  #mutate(chunk = str_pad(chunk, 3, pad = '0'), frame = str_pad(frame, 4, pad = '0'))

summary(court_coords_cleaned)
summary(abs(court_coords_cleaned$x_4 - court_coords_cleaned$x_3))
summary(abs(court_coords_cleaned$x_2 - court_coords_cleaned$x_1))
write_csv(court_coords_cleaned, '~/Documents/tennis-tracker/assets/demo/court_coords_clean.csv')
