import os

src = "tune_ett_head_dropout_position.py"
dst = "tune_nooc_Linear.py"

if os.path.exists(src):
    try:
        os.rename(src, dst)
        print(f"Successfully renamed {src} to {dst}")
    except Exception as e:
        print(f"Error renaming file: {e}")
else:
    print(f"Source file {src} not found, check if it was already renamed.")
