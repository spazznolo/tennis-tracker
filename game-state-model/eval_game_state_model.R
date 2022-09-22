


results <- read_csv('~/Documents/tennis-tracker/results.csv')

res_1 <-
  results %>%
  mutate(Filename = gsub('test/', '', Filename)) %>%
  inner_join(in_play_frames, by = c('Filename' = 'frame_id')) 

res_1 %>%
  group_by(in_play) %>% summarize(mean(Predictions))

res_1 %>%
  ggplot() +
  geom_histogram(aes(Predictions)) +
  facet_wrap(~in_play)

res_1 %>% 
  count(Predictions < 0.65, in_play)

file_names <-
  res_1 %>%
  filter(Predictions < 0.6, in_play == 'out_play') %>%
  pull(Filename)

dir.create(paste0("~/Documents/tennis-tracker/assets/view"), recursive=TRUE)

for (file_name_ in file_names) {
  
  file.copy(
    from = paste0("~/Documents/tennis-tracker/assets/test/test/", file_name_),
    to = paste0("~/Documents/tennis-tracker/assets/view/", file_name_)
  )
  
}

res_1 %>%
  mutate(frame_number = parse_number(frame)) %>%
  ggplot() +
  geom_bar(aes(frame_number, Predictions, fill = in_play), stat = 'identity') +
  facet_wrap(~chunk+game)



