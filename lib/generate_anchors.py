import numpy as np
from IPython import embed


def generate_anchors(total_stride, base_size, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)     #len返回字符串、列表、字典长度  一共5个anchor
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)  #返回指定维度的0填充数组，anchor_num行，4列，即5行四列
    size = base_size * base_size
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    # (5,4x225) to (225x5,4)
    ori = - (score_size // 2) * total_stride
    # the left displacement
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    # (15,15)
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \   #.flatten()表示把矩阵降成一维，沿横向降
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    # (15,15) to (225,1) to (5,225) to (225x5,1)
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)  #.astype()对变量进行特定类型的转换
    return anchor
