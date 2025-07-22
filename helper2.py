import os
import re
import sys

def f(directory='.'):
    pattern = re.compile(r'^file(\d{2})\.pdf$', re.IGNORECASE)
    all_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    existing_nums = []
    for fname in all_files:
        m = pattern.match(fname)
        if m:
            existing_nums.append(int(m.group(1)))
    max_num = max(existing_nums) if existing_nums else 0
    others = [f for f in sorted(all_files) if not pattern.match(f)]
    if not others:
        print("no unnumbered files.")
        return
    next_num = max_num + 1
    for old_name in others:
        new_name = f"file{next_num:02d}.pdf"
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)

        if os.path.exists(new_path):
            print(f"cant rename '{old_name}' → '{new_name}', it target already exists.")
        else:
            print(f"renaming '{old_name}' → '{new_name}'")
            os.rename(old_path, new_path)
            next_num += 1

    print("done, new max index = ", next_num - 1)

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    f(target_dir)
