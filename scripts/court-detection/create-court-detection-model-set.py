
import os
import zipfile
from collections import defaultdict

dir_path = 'assets/highlights/object-detection-train-frames'

# Get a list of all the files in the directory
files = os.listdir(dir_path)

# Use a defaultdict to group the files by their pattern (the first word in the filename)
file_groups = defaultdict(list)
for filename in files:
    pattern = filename.split()[0]
    file_groups[pattern].append(filename)

# Create a new zip file for the output
output_file = 'assets/court-detection-train-frames.zip'
with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:

    # Loop over each file group and add the first 10 files to the zip
    for pattern, group_files in file_groups.items():
        for i, filename in enumerate(group_files):
            if i >= 10:
                break
            filepath = os.path.join(dir_path, filename)
            zipf.write(filepath, arcname=filename)

# Print a message when the zip file has been created
print(f'Successfully created zip file: {output_file}')
