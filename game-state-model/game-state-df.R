
library(tidyverse)
library(stringi)

game_frames <- gsub('[.]jpg', '', list.files('~/Documents/tennis-tracker/assets/game-frames'))
split_game_frames <- stri_split_fixed(game_frames, "-", n = 5)
game_chunks <- outcome <- sub("-[^-]+$", "", game_frames)
frame_numbers <- as.numeric(sapply(split_game_frames, `[`, 5))

game_state_df <-
  tibble(
    game_chunk = game_chunks,
    frame_number = frame_numbers
    ) %>%
  arrange(game_chunk, frame_number)

game_state_df_2 <-
  game_state_df %>%
  mutate(
    game_state = case_when(
      game_chunk == 'clay-m-2009-2' & frame_number %in% c(423:432, 1065:1456) ~ 1,
      game_chunk == 'clay-m-2009-42' & frame_number %in% c(778:960) ~ 1,
      game_chunk == 'clay-m-2009-45' & frame_number %in% c(0:257, 1245:1389) ~ 1,
      game_chunk == 'clay-m-2009-53' & frame_number %in% c(0:32) ~ 1,
      game_chunk == 'clay-m-2009-59' & frame_number %in% c(213:309, 1045:1065, 1384:1600) ~ 1,
      game_chunk == 'clay-m-2009-65' & frame_number %in% c(230:268, 1083:1264) ~ 1,
      game_chunk == 'clay-m-2009-69' & frame_number %in% c(0:86) ~ 1,
      game_chunk == 'clay-m-2009-78' & frame_number %in% c(1117:1169) ~ 1,
      game_chunk == 'clay-m-2009-88' & frame_number %in% c(134:203, 1349:1396) ~ 1,
      game_chunk == 'clay-m-2009-90' & frame_number %in% c(473:581) ~ 1,
      
      
      game_chunk == 'clay-m-2012-11' & frame_number %in% c(560:815, 1315:1335) ~ 1,
      game_chunk == 'clay-m-2012-15' & frame_number %in% c(171:349, 1023:1600) ~ 1,
      game_chunk == 'clay-m-2012-22' & frame_number %in% c(384:406, 637:1019) ~ 1,
      game_chunk == 'clay-m-2012-30' & frame_number %in% c(488:509, 792:983) ~ 1,
      game_chunk == 'clay-m-2012-68' & frame_number %in% c(211:225,521:748) ~ 1,
      game_chunk == 'clay-m-2012-83' & frame_number %in% c(162:194, 664:730, 1242:1311) ~ 1,
      game_chunk == 'clay-m-2012-86' & frame_number %in% c(479:490, 690:1046) ~ 1,
      game_chunk == 'clay-m-2012-91' & frame_number %in% c(240:254, 471:855) ~ 1,
      game_chunk == 'clay-m-2012-103' & frame_number %in% c(257:290, 515:542, 1094:1236) ~ 1,
      game_chunk == 'clay-m-2012-116' & frame_number %in% c(-99) ~ 1,
      
  
      game_chunk == 'clay-w-2019-15' & frame_number %in% c(386:455, 824:913, 1301:1381) ~ 1, 
      game_chunk == 'clay-w-2019-17' & frame_number %in% c(344:470, 853:882, 1132:1331) ~ 1, 
      game_chunk == 'clay-w-2019-18' & frame_number %in% c(231:337, 731:845) ~ 1, 
      game_chunk == 'clay-w-2019-37' & frame_number %in% c(565:601, 819:878, 1247:1276) ~ 1, 
      game_chunk == 'clay-w-2019-42' & frame_number %in% c(75:94, 331:346, 785:885, 1248:1336) ~ 1, 
      game_chunk == 'clay-w-2019-45' & frame_number %in% c(1167:1354) ~ 1, 
      game_chunk == 'clay-w-2019-72' & frame_number %in% c(-99) ~ 1, 
      game_chunk == 'clay-w-2019-84' & frame_number %in% c(0:140, 640:655, 938:1111) ~ 1, 
      game_chunk == 'clay-w-2019-118' & frame_number %in% c(-99) ~ 1, 
      game_chunk == 'clay-w-2019-122' & frame_number %in% c(210:230, 495:554, 1056:1137) ~ 1,
      
      
      
      game_chunk == 'clay-w-2020-15' & frame_number %in% c(46:273, 851:950, 1482:1600) ~ 1, 
      game_chunk == 'clay-w-2020-39' & frame_number %in% c(176:280, 848:931) ~ 1, 
      game_chunk == 'clay-w-2020-45' & frame_number %in% c(-99) ~ 1, 
      game_chunk == 'clay-w-2020-49' & frame_number %in% c(74:195, 551:622, 926:1000) ~ 1, 
      game_chunk == 'clay-w-2020-52' & frame_number %in% c(138:257, 790:820, 1168:1204, 1478:1600) ~ 1, 
      game_chunk == 'clay-w-2020-79' & frame_number %in% c(139:183, 540:587, 1001:1050, 1309:1360) ~ 1, 
      game_chunk == 'clay-w-2020-93' & frame_number %in% c(0:50, 467:519, 849:878, 1082:1148) ~ 1, 
      game_chunk == 'clay-w-2020-96' & frame_number %in% c(292:500, 1130:1177, 1437:1600) ~ 1, 
      game_chunk == 'clay-w-2020-100' & frame_number %in% c(323:378, 752:865, 1192:1464) ~ 1, 
      game_chunk == 'clay-w-2020-107' & frame_number %in% c(-99) ~ 1,
      
      
      game_chunk == 'grass-m-2015-1' & frame_number %in% c(222:312) ~ 1,
      game_chunk == 'grass-m-2015-11' & frame_number %in% c(745:889, 1491:1600) ~ 1,
      game_chunk == 'grass-m-2015-43' & frame_number %in% c(-99) ~ 1,
      game_chunk == 'grass-m-2015-62' & frame_number %in% c(-99) ~ 1,
      game_chunk == 'grass-m-2015-67' & frame_number %in% c(-99) ~ 1,
      game_chunk == 'grass-m-2015-78' & frame_number %in% c(-99) ~ 1,
      game_chunk == 'grass-m-2015-113' & frame_number %in% c(122:151, 586:633, 1490:1600) ~ 1,
      game_chunk == 'grass-m-2015-119' & frame_number %in% c(-99) ~ 1,
      game_chunk == 'grass-m-2015-129' & frame_number %in% c(848:861, 1048:1149) ~ 1,
      game_chunk == 'grass-m-2015-135' & frame_number %in% c(58:72, 244:295, 1368:1600) ~ 1,
      
      
      game_chunk == 'grass-m-2019-2' & frame_number %in% c(1290:1314, 1481:1600) ~ 1,
      game_chunk == 'grass-m-2019-38' & frame_number %in% c(175:195, 1283:1498) ~ 1,
      game_chunk == 'grass-m-2019-55' & frame_number %in% c(147:170, 636:653, 1198:1322) ~ 1,
      game_chunk == 'grass-m-2019-70' & frame_number %in% c(282:345) ~ 1,
      game_chunk == 'grass-m-2019-83' & frame_number %in% c(437:517, 833:853, 1164:1195, 1366:1437) ~ 1,
      game_chunk == 'grass-m-2019-92' & frame_number %in% c(522:622) ~ 1,
      game_chunk == 'grass-m-2019-136' & frame_number %in% c(0:146, 866:896, 1206:1340) ~ 1,
      game_chunk == 'grass-m-2019-142' & frame_number %in% c(0:60, 884:960) ~ 1,
      game_chunk == 'grass-m-2019-147' & frame_number %in% c(0:29, 508:615, 1138:1204) ~ 1,
      game_chunk == 'grass-m-2019-158' & frame_number %in% c(351:380, 1345:1432) ~ 1,
   
      
      game_chunk == 'grass-w-2018-8' & frame_number %in% c(131:504, 1002:1025, 1239:1278) ~ 1,
      game_chunk == 'grass-w-2018-15' & frame_number %in% c(329:420, 1017:1044, 1205:1458) ~ 1,
      game_chunk == 'grass-w-2018-20' & frame_number %in% c(395:502, 925:1118) ~ 1,
      game_chunk == 'grass-w-2018-21' & frame_number %in% c(107:129, 363:479, 966:994, 1205:1600) ~ 1,
      game_chunk == 'grass-w-2018-26' & frame_number %in% c(320:341, 611:1381) ~ 1,
      game_chunk == 'grass-w-2018-29' & frame_number %in% c(450:549, 949:964, 1261:1278) ~ 1,
      game_chunk == 'grass-w-2018-42' & frame_number %in% c(417:553, 1035:1068, 1275:1600) ~ 1,
      game_chunk == 'grass-w-2018-49' & frame_number %in% c(81:300, 888:1224) ~ 1,
      game_chunk == 'grass-w-2018-50' & frame_number %in% c(-99) ~ 1,
      game_chunk == 'grass-w-2018-69' & frame_number %in% c(0:442, 1137:1600) ~ 1,
      
  
      
      game_chunk == 'grass-w-2019-18' & frame_number %in% c(203:454, 997:1020, 1234:1298) ~ 1,
      game_chunk == 'grass-w-2019-26' & frame_number %in% c(381:405, 614:670, 1081:1134) ~ 1,
      game_chunk == 'grass-w-2019-35' & frame_number %in% c(45:74, 265:388, 1113:1132, 1383:1445) ~ 1,
      game_chunk == 'grass-w-2019-38' & frame_number %in% c(321:385)~ 1,
      game_chunk == 'grass-w-2019-39' & frame_number %in% c(1349:1465) ~ 1,
      game_chunk == 'grass-w-2019-43' & frame_number %in% c(571:623, 1037:1065, 1388:1600) ~ 1,
      game_chunk == 'grass-w-2019-44' & frame_number %in% c(0:72, 702:713, 1005:1437) ~ 1,
      game_chunk == 'grass-w-2019-45' & frame_number %in% c(-99) ~ 1,
      game_chunk == 'grass-w-2019-48' & frame_number %in% c(235:331, 854:933, 1342:1479) ~ 1,
      game_chunk == 'grass-w-2019-57' & frame_number %in% c(86:193) ~ 1,
      
     
      
      game_chunk == 'hard-m-2019-20' & frame_number %in% c(384:400, 636:700, 1318:1800) ~ 1,
      game_chunk == 'hard-m-2019-36' & frame_number %in% c(302:401, 845:935, 1365:1394, 1596:1772) ~ 1,
      game_chunk == 'hard-m-2019-42' & frame_number %in% c(153:330, 886:1032, 1603:1693) ~ 1,
      game_chunk == 'hard-m-2019-56' & frame_number %in% c(540:587, 1116:1210) ~ 1,
      game_chunk == 'hard-m-2019-66' & frame_number %in% c(682:1221) ~ 1,
      game_chunk == 'hard-m-2019-68' & frame_number %in% c(436:737, 1515:1541, 1785:1800) ~ 1,
      game_chunk == 'hard-m-2019-85' & frame_number %in% c(649:814) ~ 1,
      game_chunk == 'hard-m-2019-103' & frame_number %in% c(629:974) ~ 1,
      game_chunk == 'hard-m-2019-124' & frame_number %in% c(755:792, 1042:1346) ~ 1,
      game_chunk == 'hard-m-2019-128' & frame_number %in% c(726:893, 1770:1800) ~ 1,
      
   
      game_chunk == 'hard-w-2017-15' & frame_number %in% c(235:538, 1106:1151) ~ 1,
      game_chunk == 'hard-w-2017-27' & frame_number %in% c(66:220, 551:587, 1028:1059, 1294:1315) ~ 1,
      game_chunk == 'hard-w-2017-29' & frame_number %in% c(141:400, 972:1024, 1354:1600) ~ 1,
      game_chunk == 'hard-w-2017-34' & frame_number %in% c(191:285, 898:971) ~ 1,
      game_chunk == 'hard-w-2017-49' & frame_number %in% c(251:295, 551:576, 1242:1438) ~ 1,
      game_chunk == 'hard-w-2017-52' & frame_number %in% c(345:373, 1370:1410) ~ 1,
      game_chunk == 'hard-w-2017-55' & frame_number %in% c(0:55, 582:692) ~ 1,
      game_chunk == 'hard-w-2017-57' & frame_number %in% c(183:235, 1282:1600) ~ 1,
      game_chunk == 'hard-w-2017-63' & frame_number %in% c(947:981, 1281:1600) ~ 1,
      game_chunk == 'hard-w-2017-70' & frame_number %in% c(845:866, 1245:1363) ~ 1,
 
      
      game_chunk == 'hard-w-2020-11' & frame_number %in% c(173:213, 798:940, 1394:1498)~ 1,
      game_chunk == 'hard-w-2020-28' & frame_number %in% c(102:196, 651:700, 894:1023)~ 1,
      game_chunk == 'hard-w-2020-36' & frame_number %in% c(176:274, 564:641, 956:1341) ~ 1,
      game_chunk == 'hard-w-2020-55' & frame_number %in% c(0:117, 829:941) ~ 1,
      game_chunk == 'hard-w-2020-57' & frame_number %in% c(302:351, 560:814, 1267:1292) ~ 1,
      game_chunk == 'hard-w-2020-58' & frame_number %in% c(0:63, 523:680, 1313:1399) ~ 1,
      game_chunk == 'hard-w-2020-61' & frame_number %in% c(1061:1273) ~ 1,
      game_chunk == 'hard-w-2020-70' & frame_number %in% c(219:516, 1177:1359) ~ 1,
      game_chunk == 'hard-w-2020-82' & frame_number %in% c(374:500, 899:985) ~ 1,
      game_chunk == 'hard-w-2020-98' & frame_number %in% c(546:611, 1056:1083, 1285:1404) ~ 1,
      
      
      game_chunk == 'hard-w-2022-8' & frame_number %in% c(0:280, 1127:1170, 1589:1700) ~ 1,
      game_chunk == 'hard-w-2022-52' & frame_number %in% c(593:991) ~ 1,
      game_chunk == 'hard-w-2022-67' & frame_number %in% c(365:434, 1309:1325) ~ 1,
      game_chunk == 'hard-w-2022-70' & frame_number %in% c(209:526, 1248:1318) ~ 1,
      game_chunk == 'hard-w-2022-82' & frame_number %in% c(-99) ~ 1,
      game_chunk == 'hard-w-2022-83' & frame_number %in% c(-99) ~ 1,
      game_chunk == 'hard-w-2022-89' & frame_number %in% c(542:561, 1041:1231) ~ 1,
      game_chunk == 'hard-w-2022-102' & frame_number %in% c(685:787, 1633:1658) ~ 1,
      game_chunk == 'hard-w-2022-108' & frame_number %in% c(-99) ~ 1,
      game_chunk == 'hard-w-2022-123' & frame_number %in% c(0:135, 1392:1408) ~ 1,
      
      
      TRUE ~ 0
      )
    ) %>%
  mutate(
    game_state = ifelse(game_state == 1, 'on', 'off'),
    model_state = case_when(
      grepl('clay-m-2009|clay-w-2019|grass-m-2015|grass-w-2018|hard-m-2019|hard-w-2020', game_chunk) ~ 'train',
      grepl('clay-m-2012|grass-m-2019|grass-w-2019|hard-w-2017', game_chunk) ~ 'val',
      TRUE ~ 'test'
    )
  )



for (i in 1:nrow(game_state_df_2)) {
  
  frame_number_ <- game_state_df_2 %>% slice(i) %>% pull(frame_number)
  game_chunk_ <- game_state_df_2 %>% slice(i) %>% pull(game_chunk)
  game_state_ <- game_state_df_2 %>% slice(i) %>% pull(game_state)
  model_state_ <- game_state_df_2 %>% slice(i) %>% pull(model_state)
  
  if (model_state_ == 'test') {
    
    file.copy(
      from = paste0("~/Documents/tennis-tracker/assets/game-frames", '/', game_chunk_, '-', frame_number_, '.jpg'),
      to = paste0('~/Documents/tennis-tracker/assets/test/test/', game_chunk_, '-', frame_number_, '.jpg')
    )
    
  } else {
    
    file.copy(
      from = paste0("~/Documents/tennis-tracker/assets/game-frames", '/', game_chunk_, '-', frame_number_, '.jpg'),
      to = paste0("~/Documents/tennis-tracker/assets/", model_state_, '/', game_state_, '/', game_chunk_, '-', frame_number_, '.jpg')
    )
    
  }
  
  
}

game_state_df_3 <-
  game_state_df_2 %>%
  filter(grepl('hard-m-2019', game_chunk), game_state == 'on') %>%
  separate(game_chunk, c('type', 'sex', 'year', 'game_chunk')) %>%
  mutate(game_chunk_l = str_pad(game_chunk, 3, pad = '0'), frame_number_l = str_pad(frame_number, 4, pad = '0'))

unlink("~/Documents/tennis-tracker/assets/demo/all-frames", recursive = TRUE)
dir.create(paste0("~/Documents/tennis-tracker/assets/demo/all-frames"), recursive=TRUE)

for (i in 1:nrow(game_state_df_3)) {
  
  frame_number_ <- game_state_df_3 %>% slice(i) %>% pull(frame_number)
  game_chunk_ <- game_state_df_3 %>% slice(i) %>% pull(game_chunk)
  frame_number_l <- game_state_df_3 %>% slice(i) %>% pull(frame_number_l)
  game_chunk_l <- game_state_df_3 %>% slice(i) %>% pull(game_chunk_l)
  
  file.copy(
    from = paste0("~/Documents/tennis-tracker/assets/game-frames/hard-m-2019-", game_chunk_, '-', frame_number_, '.jpg'),
    to = paste0("~/Documents/tennis-tracker/assets/demo/all-frames/hard-m-2019-", game_chunk_l, '-', frame_number_l, '.jpg')
  )
  
}




