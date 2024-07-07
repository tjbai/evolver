#!/bin/bash

# Directory containing the files
dir="/scratch4/jeisner1/cache"

# Change to the directory
cd "$dir" || exit

# Rename the files
for file in sup-ud-3.0_*.zst; do
    if [ -e "$file" ]; then
        new_name=$(echo "$file" | sed 's/sup-ud-3.0_/sup-ud-3_/')
        mv "$file" "$new_name"
        echo "Renamed $file to $new_name"
    fi
done

echo "Renaming complete!"
