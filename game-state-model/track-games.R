

library(tidyverse)

tibble(
  game-chunk = 'clay-m-2012'
)

clay-m-2012-0 <- rep(0, 1600)

clay-m-2012-17 <- rep(0, 1600)
clay-m-2012-17[c(821:936)] <- 1

clay-m-2012-25 <- rep(0, 1600)
clay-m-2012-25[c(211:249, 1205:1216, 1395:1444)] <- 1

clay-m-2012-71 <- rep(0, 1600)
clay-m-2012-71[c(179:1155)] <- 1

clay-m-2012-79 <- rep(0, 1600)
clay-m-2012-79[c(386:471, 1273:1387)] <- 1

clay-m-2012-80 <- rep(0, 1600)
clay-m-2012-80[c(374:779, 1343:1600)] <- 1

clay-m-2012-99 <- rep(0, 1600)
clay-m-2012-99[c(242:254, 452:465, 996:1074)] <- 1

clay-m-2012-111 <- rep(0, 1600)
clay-m-2012-111[c(0:226, 1163:1178, 1469:1600)] <- 1

clay-m-2012-112 <- rep(0, 1600)
clay-m-2012-112[c(0:96, 1006:1052)] <- 1

clay-m-2012-124 <- rep(0, 1600)





















in_play_set <-
  tibble(
    game = 2,
    chunk = unlist(map(20:25, ~rep(paste0('chunk_', .), 1600))),
    frame = rep(paste0('frame_', 1:1600, '.jpg'), 6),
    in_play = c(chunk_20, chunk_21, chunk_22, chunk_23, chunk_24, chunk_25)
  )

states <- c('train', 'val', 'test')

in_play_frames <-
  in_play_set %>%
  mutate(
    in_play = ifelse(in_play == 1, 'in_play', 'out_play'),
    frame_id = paste0('_', game, '_', parse_number(chunk), '_', frame)
  )

in_play_frames$group <- sample(c('train', 'train', 'val'), nrow(in_play_frames), replace = TRUE)

in_play_frames <-
  in_play_frames %>% 
  mutate(group = ifelse(chunk %in% paste0('chunk_', 24:25), 'test', group))

for (i in 1:nrow(in_play_frames)) {
  
  frame_p <- in_play_frames %>% slice(i) %>% pull(frame)
  frame_l <- in_play_frames %>% slice(i) %>% pull(frame_id)
  frame_n <- in_play_frames %>% slice(i) %>% pull(in_play)
  frame_m <- in_play_frames %>% slice(i) %>% pull(group)
  frame_x <- in_play_frames %>% slice(i) %>% pull(chunk)
  frame_y <- in_play_frames %>% slice(i) %>% pull(game)
  
  if (frame_m == 'test') {
    
    file.copy(
      from = paste0("~/Documents/tennis-tracker/assets/game_", frame_y, '/', frame_x, "/", frame_p),
      to = paste0("~/Documents/tennis-tracker/assets/", frame_m, "/test/", frame_l)
    )
    
  } else {
    
    file.copy(
      from = paste0("~/Documents/tennis-tracker/assets/game_", frame_y, '/', frame_x, "/", frame_p),
      to = paste0("~/Documents/tennis-tracker/assets/", frame_m, "/", frame_n, "/", frame_l)
    )
    
  }
  
  
}
