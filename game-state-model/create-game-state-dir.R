
unlink("~/Documents/tennis-tracker/assets/train/on", recursive = TRUE)
unlink("~/Documents/tennis-tracker/assets/train/off", recursive = TRUE)
unlink("~/Documents/tennis-tracker/assets/val/on", recursive = TRUE)
unlink("~/Documents/tennis-tracker/assets/val/off", recursive = TRUE)
unlink("~/Documents/tennis-tracker/assets/test/test", recursive = TRUE)

dir.create(paste0("~/Documents/tennis-tracker/assets/train/on"), recursive=TRUE)
dir.create(paste0("~/Documents/tennis-tracker/assets/train/off"), recursive=TRUE)
dir.create(paste0("~/Documents/tennis-tracker/assets/val/on"), recursive=TRUE)
dir.create(paste0("~/Documents/tennis-tracker/assets/val/off"), recursive=TRUE)
dir.create(paste0("~/Documents/tennis-tracker/assets/test/test"), recursive=TRUE)