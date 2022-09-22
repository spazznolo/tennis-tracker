
library(tidyverse)

chunk_0 <- rep(0, 1600)
chunk_0[c(592:605, 1052:1191)] <- 1

chunk_1 <- rep(0, 1600)
chunk_1[c(204:280, 746:757, 1191:1199, 1465:1506)] <- 1

chunk_2 <- rep(0, 1600)
chunk_2[c(442:571)] <- 1

chunk_3 <- rep(0, 1600)
chunk_3[c(238:251, 531:570, 972:1113)] <- 1

chunk_4 <- rep(0, 1600)
chunk_4[c(202:215, 513:522, 1005:1012, 1333:1499)] <- 1

chunk_5 <- rep(0, 1600)
chunk_5[c(429:448, 946:953, 1304:1600)] <- 1

chunk_6 <- rep(0, 1600)
chunk_6[c(0:136, 693:794)] <- 1

chunk_7 <- rep(0, 1600)
chunk_7[c(108:116, 412:550, 1405:1513)] <- 1

chunk_8 <- rep(0, 1600)
chunk_8[c(605:613, 856:895, 1372:1379)] <- 1

chunk_9 <- rep(0, 1600)
chunk_9[c(112:122, 494:576, 1087:1093, 1375:1600)] <- 1

chunk_10 <- rep(0, 1600)
chunk_10[c(0:205, 984:1123)] <- 1

chunk_11 <- rep(0, 1600)
chunk_11[c(464:475, 795:987)] <- 1

chunk_12 <- rep(0, 1600)
chunk_12[c(718:724, 992:1068, 1554:1600)] <- 1

chunk_13 <- rep(0, 1600)
chunk_13[c(0:119, 582:651, 1186:1250)] <- 1

chunk_14 <- rep(0, 1600)
chunk_14[c(200:212, 440:581, 1488:1498)] <- 1

chunk_15 <- rep(0, 1600)
chunk_15[c(286:571, 1157:1176)] <- 1

chunk_16 <- rep(0, 1600)
chunk_16[c(251:257, 631:643, 955:1016, 1502:1600)] <- 1

chunk_17 <- rep(0, 1600)
chunk_17[c(0:117, 606:638, 1185:1196, 1500:1509)] <- 1

chunk_18 <- rep(0, 1600)
chunk_18[c(554:564, 847:857, 1234:1253)] <- 1

chunk_19 <- rep(0, 1600)
chunk_19[c(221:248, 580:590, 876:945, 1339:1349)] <- 1

in_play_set <-
  tibble(
    game = 1,
    chunk = unlist(map(0:19, ~rep(paste0('chunk_', .), 1600))),
    frame = rep(paste0('frame_', 1:1600, '.jpg'), 20),
    in_play = c(chunk_0, chunk_1, chunk_2, chunk_3, chunk_4, chunk_5, chunk_6, chunk_7, chunk_8, chunk_9,
                chunk_10, chunk_11, chunk_12, chunk_13, chunk_14, chunk_15, chunk_16, chunk_17, chunk_18, chunk_19)
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
  filter(chunk %in% paste0('chunk_', c(5:10, 16:19))) %>%
  mutate(group = ifelse(chunk %in% paste0('chunk_', 16:19), 'test', group))

for (i in 1:nrow(in_play_frames)) {

  frame_p <- in_play_frames %>% slice(i) %>% pull(frame)
  frame_l <- in_play_frames %>% slice(i) %>% pull(frame_id)
  frame_n <- in_play_frames %>% slice(i) %>% pull(in_play)
  frame_m <- in_play_frames %>% slice(i) %>% pull(group)
  frame_x <- in_play_frames %>% slice(i) %>% pull(chunk)
  
  if (frame_m == 'test') {
    
    file.copy(
      from = paste0("~/Documents/tennis-tracker/assets/", frame_x, "/", frame_p),
      to = paste0("~/Documents/tennis-tracker/assets/", frame_m, "/test/", frame_l)
    )
    
  } else {
    
    file.copy(
      from = paste0("~/Documents/tennis-tracker/assets/", frame_x, "/", frame_p),
      to = paste0("~/Documents/tennis-tracker/assets/", frame_m, "/", frame_n, "/", frame_l)
    )
    
  }

  
}
