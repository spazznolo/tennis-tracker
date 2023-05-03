

library(tidyverse)
library(zoo)
library(tools)

event_prediction_output <- 
  read_csv('~/Downloads/X_predictions (4).csv') %>%
  group_by(match_id) %>%
  mutate(imputed_mean = rollmean(ball_imputed, k = 10, align = 'center', na.pad = TRUE)) %>%
  ungroup()

bounce_hit_file_paths <- list.files('~/Documents/tennis-tracker/assets/highlights/bounce-hit-frames', full.names = TRUE)

hit_db <- map_dfr(bounce_hit_file_paths, ~{
  file_name <- file_path_sans_ext(basename(.x))
  data <- read_csv(.x)
  data$match_id <- file_name
  return(data)
})

imputed_dict <-
  event_prediction_output %>%
  filter(imputed_mean > 0.9) %>%
  filter(frame > lag(frame) + 1) %>%
  select(match_id, frame) %>%
  mutate(example = 1:n())

filtered_df <- 
  event_prediction_output %>%
  inner_join(imputed_dict, by = "match_id") %>%
  filter(abs(frame.x - frame.y) <= 25) %>%
  select(-frame.y) %>%
  rename(frame = frame.x)

filtered_df %>%
  filter(example < 10) %>%
  ggplot(aes(ball_x_imputed, -ball_y_imputed, col = ball_imputed)) +
  geom_point() +
  geom_path() +
  facet_wrap(~example)

misclasses <-
  event_prediction_output %>%
  filter(ball_imputed == 0) %>%
  count(match_id, ball_x_imputed, ball_y_imputed, sort = TRUE)

event_prediction_output %>% 
  filter(grepl('Auckland', match_id), ball_x_imputed == 0.654, ball_y_imputed == 0.129) 

event_prediction_output %>% 
  filter(grepl('Auckland', match_id)) %>%
  mutate(ball_speed = sqrt((ball_x_imputed - lag(ball_x_imputed))^2 + (ball_y_imputed - lag(ball_y_imputed))^2)) %>%
  select(match_id, frame, ball_x_imputed, ball_y_imputed, ball_imputed, ball_speed) %>%
  filter(frame == lag(frame) + 1) %>%
  ggplot() +
  geom_histogram(aes(ball_speed))

event_prediction_output %>% 
  filter(grepl('Auckland', match_id)) %>%
  mutate(ball_speed = sqrt((ball_x_imputed - lag(ball_x_imputed))^2 + (ball_y_imputed - lag(ball_y_imputed))^2)) %>%
  select(match_id, frame, ball_x_imputed, ball_y_imputed, ball_imputed, ball_speed) %>%
  filter(frame == lag(frame) + 1) %>%
  pull(ball_speed) %>%
  quantile(., probs = seq(0, 1, 0.05))

event_predictions <-
  event_prediction_output %>%
  mutate(
    across(c(bounce_ewm, hit_ewm), ~round(., 2)),
    bounce_binary = ifelse(bounce_ewm <= 12 & bounce_ewm <= lag(bounce_ewm, 1) & bounce_ewm <= lead(bounce_ewm, 1) &
                          bounce_ewm <= lag(bounce_ewm, 5) & bounce_ewm <= lead(bounce_ewm, 5), 1, 0),
    hit_binary = ifelse(hit_ewm <= 12 & hit_ewm <= lag(hit_ewm, 1) & hit_ewm <= lead(hit_ewm, 1) &
                       hit_ewm <= lag(hit_ewm, 5) & hit_ewm <= lead(hit_ewm, 5), 1, 0)) %>%
  group_by(match_id) %>%
  mutate(
    point = ifelse(frame - 50 > lag(frame), 1, 0),
    point = replace_na(point, 0),
    point = cumsum(point)) %>%
  ungroup() %>%
  filter(bounce_binary == 1 | hit_binary == 1) %>%
  select(match_id, point, frame, bounce_binary, bounce_ewm, hit_binary, hit_ewm, ball_x_imputed, ball_y_imputed, ball_imputed) %>%
  arrange(match_id, frame, point) 

  


hit_db_events <-
  hit_db %>%
  filter(event != 4) %>%
  group_by(match_id) %>%
  mutate(
    point = ifelse(frame - 50 > lag(frame), 1, 0),
    point = replace_na(point, 0),
    point = cumsum(point)) %>%
  ungroup()

hit_db_start_points <-
  hit_db_events %>%
  group_by(match_id, point) %>%
  slice(1) %>%
  ungroup()

event_predictions_start_points <-
  event_predictions %>% 
  select(-point) %>%
  left_join(hit_db_start_points %>% select(match_id, true_frame = frame, point), by = c('match_id')) %>%
  filter(frame >= true_frame - 5) %>%
  mutate(diff = abs(frame - true_frame)) %>%
  arrange(match_id, frame, diff) %>%
  distinct(match_id, frame, .keep_all = TRUE) %>%
  count(match_id, point, name = 'predicted_events')

comp <-
  hit_db_events %>%
  count(match_id, point, name = 'actual_events') %>%
  left_join(event_predictions_start_points, by = c('match_id', 'point')) %>%
  mutate(delta = actual_events - predicted_events)

event_predictions_comp <-
  event_predictions %>% 
  select(-point) %>%
  left_join(hit_db_start_points %>% select(match_id, true_frame = frame, point), by = c('match_id')) %>%
  filter(frame >= true_frame - 5) %>%
  mutate(diff = abs(frame - true_frame)) %>%
  arrange(match_id, frame, diff) %>%
  distinct(match_id, frame, .keep_all = TRUE) %>%
  select(match_id, point, frame, ball_x_imputed, ball_y_imputed, ball_imputed, bounce_binary, hit_binary, hit_ewm, bounce_ewm) %>%
  group_by(point) %>%
  filter(
    is.na(lag(bounce_ewm)) |
      !((bounce_binary == 1 & lag(bounce_binary) == 1 & bounce_ewm >= lag(bounce_ewm)) |
        (bounce_binary == 1 & lead(bounce_binary) == 1 & bounce_ewm > lead(bounce_ewm)) |
        (hit_binary == 1 & lag(hit_binary) == 1 & hit_ewm >= lag(hit_ewm) & frame - lag(frame) <= 10) |
        (hit_binary == 1 & lead(hit_binary) == 1 & hit_ewm > lead(hit_ewm) & lead(frame) - frame <= 10))
  ) %>%
  ungroup()

come_on <-
  hit_db_events %>%
  group_by(match_id, point) %>%
  mutate(event_number = 1:n()) %>%
  ungroup() %>%
  full_join(event_predictions_comp %>%
              filter(!(frame == lag(frame) + 1)) %>%
              group_by(match_id, point) %>%
              mutate(event_number = 1:n()) %>%
              ungroup(), by = c('match_id', 'point', 'event_number'))

point_hit_rate <-
  come_on %>%
  mutate(
    diff = abs(frame.x - frame.y),
    error = ifelse(diff > 5, 1, 0)) %>%
  group_by(match_id, point) %>%
  summarize(
    events = n(),
    avg_diff = mean(diff, na.rm = TRUE),
    med_diff = median(diff, na.rm = TRUE),
    error_rate = mean(error, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  filter(point > 0, events > 1)

point_hit_rate %>%
  summarize(across(avg_diff:error_rate, ~quantile(., probs = seq(0, 1, 0.1), na.rm =TRUE)))

event_prediction_output %>%
  filter(match_id == 'Auckland-2023-Quarter-Final-Highlights', frame > 1720, frame < 1750) %>%
  ggplot() +
  geom_point(aes(ball_x_imputed, ball_y_imputed, alpha = frame))

come_on %>%
  filter(match_id == 'Auckland-2023-Quarter-Final-Highlights') %>%
  ggplot() +
  geom_point(aes(ball_x_imputed, -ball_y_imputed, col = as.factor(hit_binary))) +
  facet_wrap(~point)


