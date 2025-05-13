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
    - One section for 'V' orientation, one for 'H' orientation per base name.
    - First writes a summary table with Base, V-image, H-image columns.
    - Then writes the detailed Ref X / Ref Y rows for each base.
    """
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Build mapping: base_name -> {'V': {...}, 'H': {...}}
    paired_data = {}
    for image_stem, values in data_dict.items():
        # split on last underscore to detect orientation suffix
        parts = image_stem.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ('V', 'H'):
            base, orientation = parts
        else:
            # skip any stems without _V or _H
            continue
        paired_data.setdefault(base, {})[orientation] = {
            'X': ensure_list(values.get('X')),
            'Y': ensure_list(values.get('Y'))
        }

    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # --- Summary section ---
        writer.writerow(['Base', 'V Image', 'H Image'])
        for base, pair in paired_data.items():
            v_img = f"{base}_V" if 'V' in pair else ''
            h_img = f"{base}_H" if 'H' in pair else ''
            writer.writerow([base, v_img, h_img])
        writer.writerow([])  # blank line

        # --- Detailed OCR section ---
        for base, pair in paired_data.items():
            v_data = pair.get('V', {'X': [], 'Y': []})
            h_data = pair.get('H', {'X': [], 'Y': []})

            # Pad lists to max length for symmetry
            max_len_x = max(len(v_data['X']), len(h_data['X']))
            max_len_y = max(len(v_data['Y']), len(h_data['Y']))
            v_data['X'] += [""] * (max_len_x - len(v_data['X']))
            h_data['X'] += [""] * (max_len_x - len(h_data['X']))
            v_data['Y'] += [""] * (max_len_y - len(v_data['Y']))
            h_data['Y'] += [""] * (max_len_y - len(h_data['Y']))

            # Header row with image stems
            writer.writerow([f"{base}_V"] + [""] * 7 + [f"{base}_H"])

            # Ref X row
            writer.writerow(
                ["Ref X:"] +
                v_data['X'] + [""] * (7 - len(v_data['X'])) +
                ["Ref X:"] + h_data['X']
            )

            # Ref Y row
            writer.writerow(
                ["Ref Y:"] +
                v_data['Y'] + [""] * (7 - len(v_data['Y'])) +
                ["Ref Y:"] + h_data['Y']
            )

            writer.writerow([])  # spacer row
