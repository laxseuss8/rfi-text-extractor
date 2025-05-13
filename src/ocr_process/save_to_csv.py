import csv
import os

def ensure_list(val):
    """Ensure the value is a list."""
    if isinstance(val, list):
        return val
    elif val is None:
        return []
    else:
        return [val]

def save_side_by_side_csv(data_dict, output_filename):
    """
    Save OCR data in side-by-side comparison format:
    - One column for the image stem,
    - One column per X value,
    - One column per Y value,
    - Blank line separating each image.
    """
    # make sure output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(["Image Stem", "Ref X", "Ref Y"])

        # For each image, write all X/Y pairs
        for stem, values in data_dict.items():
            x_list = ensure_list(values.get('X'))
            y_list = ensure_list(values.get('Y'))

            # pad so both lists are equal length
            max_len = max(len(x_list), len(y_list))
            x_list += [""] * (max_len - len(x_list))
            y_list += [""] * (max_len - len(y_list))

            # write each pair on its own row
            for i in range(max_len):
                writer.writerow([stem, x_list[i], y_list[i]])

            # spacer row between images
            writer.writerow([])
