import random
# 增加数据的多样性，提高模型的泛化能力

# 对图像和目标(ground truth)进行随机的水平翻转、垂直翻转等数据增强操作
def augment(img, gt, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5  # 50%概率水平翻转
    vflip = rot and random.random() < 0.5    # 50%概率垂直翻转

    if hflip: # 执行水平翻转
        img = img[:, ::-1, :].copy() # ::-1是切片操作，从末尾向开头反向选择  copy确保返回的是数据的副本
        gt = gt[:, ::-1, :].copy()
    if vflip: # 执行垂直翻转
        img = img[::-1, :, :].copy()
        gt = gt[::-1, :, :].copy()

    return img, gt

# 从图像和gt中随机裁剪出指定大小的补丁（局部区域）
def get_patch(img, gt, patch_size=16):
    th, tw = img.shape[:2]  # 图像的高度和宽度

    tp = round(patch_size)  # 将patch_size四舍五入到最近的整数 因为像素坐标必须是整数

    # 随机选择补丁的起始坐标tx,ty 确保不会超出图像边界
    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[ty:ty + tp, tx:tx + tp, :], gt[ty:ty + tp, tx:tx + tp, :]