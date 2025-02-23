def get_grid_location(cx, cy, width, height):
    row = "top" if cy < height / 3 else "center" if cy < 2 * height / 3 else "bottom"
    col = "left" if cx < width / 3 else "center" if cx < 2 * width / 3 else "right"
    return "center" if row == "center" and col == "center" else f"{row} {col}"

def remove_file_with_retry(filepath, retries=5, delay=1.0):
    import os, time
    for _ in range(retries):
        try:
            os.remove(filepath)
            return
        except Exception:
            time.sleep(delay)
