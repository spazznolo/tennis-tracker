import os
import shutil

# Define the directories
initial_directory = "assets/tennis-tracker.v10i.yolov5pytorch"
replacement_directory = "assets/highlights/object-detection-frame-differences"

# Loop through all directories and files in the initial directory
for root, directories, files in os.walk(initial_directory):
    for file_name in files:
        # Get the full file path of the initial file
        initial_file_path = os.path.join(root, file_name)
        
        # Get the full file path of the replacement file
        replacement_file_path = os.path.join(replacement_directory, file_name)
        
        # Check if the replacement file exists
        if os.path.exists(replacement_file_path):
            # Replace the initial file with the replacement file
            shutil.copy2(replacement_file_path, initial_file_path)
