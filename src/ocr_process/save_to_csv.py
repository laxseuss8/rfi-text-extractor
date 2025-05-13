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
    - Cells are laid out exactly as in your specification.
    """
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Build a mapping: base_name -> {'V': {...}, 'H': {...}}
    paired_data = {}
    for image_stem, values in data_dict.items():
        # detect a trailing " V" or " H"
        parts = image_stem.rsplit(' ', 1)
        if len(parts) == 2 and parts[1] in ('V', 'H'):
            base, orientation = parts
        else:
            # no suffix: treat it as 'V' by default
            base, orientation = image_stem, 'V'
        paired_data.setdefault(base, {})[orientation] = {
            'X': ensure_list(values.get('X')),
            'Y': ensure_list(values.get('Y'))
        }

    # Write to CSV
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        for base, pair in paired_data.items():
            v_data = pair.get('V', {'X': [], 'Y': []})
            h_data = pair.get('H', {'X': [], 'Y': []})

            # Pad both lists to max length for symmetry
            max_len_x = max(len(v_data['X']), len(h_data['X']))
            max_len_y = max(len(v_data['Y']), len(h_data['Y']))

            v_data['X'] += [""] * (max_len_x - len(v_data['X']))
            h_data['X'] += [""] * (max_len_x - len(h_data['X']))
            v_data['Y'] += [""] * (max_len_y - len(v_data['Y']))
            h_data['Y'] += [""] * (max_len_y - len(h_data['Y']))

            # Header row with image stems
            row1 = [f"{base} V"] + [""] * 7 + [f"{base} H"]
            writer.writerow(row1)

            # Ref X row
            row2 = ["Ref X:"] \
                 + v_data['X'] + [""] * (7 - len(v_data['X'])) \
                 + ["Ref X:"] + h_data['X']
            writer.writerow(row2)

            # Ref Y row
            row3 = ["Ref Y:"] \
                 + v_data['Y'] + [""] * (7 - len(v_data['Y'])) \
                 + ["Ref Y:"] + h_data['Y']
            writer.writerow(row3)

            # Spacer
            writer.writerow([])
