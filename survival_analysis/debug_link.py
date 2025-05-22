import os

# 路径设置
link_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_KIRC_directory_copy"
link_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_KIRC_directory"
link_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_directory_copy"
link_dir = "/ailab/public/pjlab-smarthealth03/leiwenhui/YushengTan/survival_analysis/TCGA_BLCA_directory"
# 初始化计数器
total_links = 0
broken_links = 0
valid_links = 0
non_symlinks = 0

# 遍历 link_dir 中的所有文件
for svs_file in os.listdir(link_dir):
    if not svs_file.endswith('.svs'):
        continue

    total_links += 1
    link_path = os.path.join(link_dir, svs_file)

    # 检查是否是软链接
    if not os.path.islink(link_path):
        print(f"⚠️ Not a symlink: {svs_file}")
        non_symlinks += 1
        continue

    # 检查软链接是否指向有效路径
    target_path = os.readlink(link_path)  # 获取软链接目标路径（可能是相对路径）
    if not os.path.isabs(target_path):    # 如果是相对路径，转为绝对路径
        target_path = os.path.join(os.path.dirname(link_path), target_path)

    if not os.path.exists(target_path):
        print(f"❌ Broken symlink: {svs_file} → {target_path}")
        broken_links += 1
    else:
        valid_links += 1

# 打印汇总
print("\n📊 Symlink Check Summary:")
print(f"🔗 Total .svs links checked: {total_links}")
print(f"✅ Valid links: {valid_links}")
print(f"❌ Broken links: {broken_links}")
print(f"⚠️ Non-symlink .svs files: {non_symlinks}")
