import os

def write_paired_png_paths(gt_folder, lq_folder, gt_txt, lq_txt):
    gt_paths = []
    lq_paths = []
    missing = []

    for root, dirs, files in os.walk(gt_folder):
        dirs.sort()
        for file in sorted(files):
            if not file.endswith('.png'):
                continue
            gt_path = os.path.join(root, file)
            rel = os.path.relpath(gt_path, gt_folder)
            lq_path = os.path.join(lq_folder, rel)
            if not os.path.exists(lq_path):
                missing.append(rel)
                continue
            gt_paths.append(gt_path)
            lq_paths.append(lq_path)

    if missing:
        print(f"WARNING: {len(missing)} GT files have no matching LQ. First few:")
        for m in missing[:5]:
            print(f"  {m}")

    with open(gt_txt, 'w') as f:
        f.write('\n'.join(gt_paths) + '\n')
    with open(lq_txt, 'w') as f:
        f.write('\n'.join(lq_paths) + '\n')

    print(f"Wrote {len(gt_paths)} paired samples")


gt_folder = '/kaggle/input/datasets/vende14/hq-lq-netherlands-train/hq_images/hq_images'
lq_folder = '/kaggle/input/datasets/vende14/hq-lq-netherlands-train/lq_images/lq_images'
write_paired_png_paths(gt_folder, lq_folder, '/gt_path.txt', '/lq_path.txt')
