#!/bin/bash
# Usage: ./count_classes.sh <label_directory>
# Example: ./count_classes.sh /part/data_fat32/xViewDataset_tar/labels

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <label_directory>"
    exit 1
fi

label_dir="$1"

if [ ! -d "$label_dir" ]; then
    echo "Error: $label_dir is not a directory."
    exit 1
fi

# Create a temporary file to store the extracted class ids
temp_file=$(mktemp)

total_lines=0

# Process all .txt files in the directory (non-recursive)
# If you want to process subdirectories as well, remove the -maxdepth 1 flag.
for file in "$label_dir"/*.txt; do
    if [ -f "$file" ]; then
        # Count non-blank lines in the file and add to total_lines.
        count=$(grep -cve '^\s*$' "$file")
        total_lines=$((total_lines + count))
        # Extract the first field of each line, convert it to int (e.g., "15.0" becomes "15")
        # and append to the temporary file.
        awk '{printf "%d\n", $1}' "$file" >> "$temp_file"
    fi
done

# If no .txt files were found, warn the user.
if [ ! -s "$temp_file" ]; then
    echo "No label lines found in $label_dir."
    rm "$temp_file"
    exit 0
fi

# Sort the class ids and remove duplicates
unique_classes=$(sort -n "$temp_file" | uniq)
unique_count=$(echo "$unique_classes" | wc -l)

# Format the unique classes into a comma-separated list.
# This replaces newlines with commas and strips the trailing comma.
class_list=$(echo "$unique_classes" | paste -sd ',' -)

echo "✅ Total unique classes: $unique_count"
echo "📋 Class list: [$class_list]"
echo "🧪 Total lines parsed: $total_lines"

# Clean up the temporary file.
rm "$temp_file"

