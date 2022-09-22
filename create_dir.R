
unlink("~/Documents/tennis-tracker/assets/train/in_play", recursive = TRUE)
unlink("~/Documents/tennis-tracker/assets/train/out_play", recursive = TRUE)
unlink("~/Documents/tennis-tracker/assets/val/in_play", recursive = TRUE)
unlink("~/Documents/tennis-tracker/assets/val/out_play", recursive = TRUE)
unlink("~/Documents/tennis-tracker/assets/test/test", recursive = TRUE)

dir.create(paste0("~/Documents/tennis-tracker/assets/train/in_play"), recursive=TRUE)
dir.create(paste0("~/Documents/tennis-tracker/assets/train/out_play"), recursive=TRUE)
dir.create(paste0("~/Documents/tennis-tracker/assets/val/in_play"), recursive=TRUE)
dir.create(paste0("~/Documents/tennis-tracker/assets/val/out_play"), recursive=TRUE)
dir.create(paste0("~/Documents/tennis-tracker/assets/test/test"), recursive=TRUE)