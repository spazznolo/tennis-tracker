
library(tidyverse)

houghs <- 
  read_csv('~/Documents/tennis-tracker/assets/temp/hough_lines.csv')

lengthwise <-
  houghs %>%
  filter(!is.na(clust)) 

widthwise <-
  houghs %>%
  filter(abs(slope) < 0.1)

widtwise %>%
  ggplot() +
  geom_histogram(aes(y_min)) +
  facet_wrap(~image)

houghs %>% 
  #filter(grepl('hard', image)) %>%
  count(image)
      
start_ = Sys.time()  

for (i in 1:26) {
  
  dist_mat <- dist(houghs %>% filter(grepl('grass-w-2018-49-223.jpg', image)) %>% select(abs_slope), method = 'euclidean')
  hclust_avg <- hclust(dist_mat, method = 'average')
  clusts <- cutree(hclust_avg, h = 0.15)
  clusts
  plot(hclust_avg)
  houghs %>% filter(grepl('grass-w-2018-49-1054.jpg', image)) 
  
}

Sys.time() - start_

length(list.files('~/Documents/tennis-tracker/assets/train/on'))

install.packages('PlaneGeometry')
library(PlaneGeometry)

line1 <- Line$new(A = c(626, 293), B = c(663,387))
line2 <- Line$new(A = c(0,2), B = c(4,2))
intersectionLineLine(line1, line2)


houghs %>% 
  filter(grepl('clay-m-2009-45-250.jpg', image), abs_slope < 0.1) %>%
  ggplot() +
  geom_histogram(aes(y_min))






houghs %>% 
  select(-1) %>%
  mutate(idx = 1:n()) %>%
  filter(image == 'grass-w-2018-42-1496.jpg', abs(slope) < 1.75, abs(slope) > 1.5)


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
