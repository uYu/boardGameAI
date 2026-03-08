import numpy as np

class PatchworkEncoder:
    def __init__(self):
        self.patches = []

    def add(self, cost, time, income, shape_str, name=""):
        # 清理字符串并转为 numpy 矩阵
        lines = [line.strip() for line in shape_str.strip().split('\n') if line.strip()]
        if not lines: return
        h = len(lines)
        w = max(len(line) for line in lines)
        matrix = np.zeros((h, w), dtype=int)
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == '#':
                    matrix[r, c] = 1
        
        self.patches.append({
            "cost": cost, "time": time, "income": income, 
            "matrix": matrix, "name": name
        })

    def generate_cpp_header(self):
        header = []
        header.append("#pragma once\n")
        header.append("#include <vector>\n#include <cstdint>\n")
        header.append("// 确保这里定义了 uint128，如果 PatchworkGame.hpp 已经定义了，可以注释掉这行")
        header.append("typedef __int128 uint128;\n")
        header.append("struct PieceData {\n    int id;\n    int cost;\n    int time;\n    int income;\n    std::vector<uint128> all_legal_masks;\n};\n")
        header.append("const std::vector<PieceData> ALL_PIECES = {")

        for i, p in enumerate(self.patches):
            unique_masks = self._get_all_bitmasks(p['matrix'])
            mask_strs = []
            for m in unique_masks:
                low = m & 0xFFFFFFFFFFFFFFFF
                high = m >> 64
                mask_strs.append(f"((uint128){high}ULL << 64 | {low}ULL)")
            
            masks_joined = ", ".join(mask_strs)
            # 修正点：去掉元组括号，使用标准 C++ 初始化列表格式
            line = f"    {{ {i}, {p['cost']}, {p['time']}, {p['income']}, {{ {masks_joined} }} }}"
            if i < len(self.patches) - 1:
                line += ","
            header.append(f"{line} // {p['name']}")
        
        header.append("};")
        return "\n".join(header)

    def _get_all_bitmasks(self, shape):
        forms = []
        curr = shape
        for _ in range(4): # 旋转
            curr = np.rot90(curr)
            forms.append(curr)
            forms.append(np.fliplr(curr)) # 翻转
        
        unique_forms = []
        for f in forms:
            if not any(np.array_equal(f, u) for u in unique_forms):
                unique_forms.append(f)

        masks = []
        for form in unique_forms:
            h, w = form.shape
            # 9x9 棋盘，确保不越界
            for r in range(9 - h + 1): 
                for c in range(9 - w + 1):
                    m = 0
                    for dr in range(h):
                        for dc in range(w):
                            if form[dr, dc]:
                                m |= (int(1) << ((r + dr) * 9 + (c + dc)))
                    masks.append(m)
        return list(set(masks)) # 去重

# --- 汇总全拼块数据 ---
encoder = PatchworkEncoder()

# 这里保留你原来的 add 调用...

# 1-3 图1
# 1. 灰色 1x2 小条
encoder.add(2, 1, 0, "##", "Grey_1x2")

# 2. 蓝色 C 形 (5格高)
encoder.add(1, 5, 1, """
##
#.
#.
#.
##
""", "Blue_C")

# 3. 黄色 L 型 (带转角)
encoder.add(3, 2, 1, """
.#
##
#..
""", "Yellow_L_stair")

# 4-9图2
# 4. 深蓝 U 型
encoder.add(1, 2, 0, """
#.#
###
""", "DarkBlue_U")

# 5. 浅蓝 H 型
encoder.add(2, 3, 0, """
#.#
###
#.#
""", "LightBlue_H")

# 6. 紫花 T 型 (长柄)
encoder.add(2, 2, 0, """
.#.
###
.#.
""", "Purple_Cross") # 实际是十字形

# 7. 蓝花 1x3 长条
encoder.add(2, 2, 0, "###", "BlueFlower_1x3")

# 8. 粉花 L 型 (最小)
encoder.add(3, 1, 0, """
#.
##
""", "Pink_L_small")

# 9. 橙色正方形 (2x2)
encoder.add(6, 5, 2, """
##
##
""", "Orange_Square")

# 10-15 图3
# 1. 浅色 T 型 (短柄)
# 消耗: 3, 时间: 4, 收入: 1
encoder.add(3, 4, 1, """
.#..
####
""", "Light_T_short")

# 2. 绿色 1x4 带侧钩 (倒 T 长柄)
# 消耗: 7, 时间: 2, 收入: 2
encoder.add(7, 2, 2, """
.#.
.#.
.#.
###
""", "Green_Long_L") # 形状像长 L

# 3. 黄色 L 型 (长臂)
# 消耗: 9, 时间: 3, 收入: 2
encoder.add(10, 3, 2, """
.#
.#
.#
##
""", "Yellow_L_long")

# 4. 灰白色 凸字型 (宽底)
# 消耗: 7, 时间: 4, 收入: 2
encoder.add(7, 4, 2, """
.##.
####
""", "White_Broad_T")

# 5. 深绿色 十字型 (带一长臂)
# 消耗: 1, 时间: 4, 收入: 1
encoder.add(1, 4, 1, """
..#..
#####
..#..
""", "DarkGreen_Cross_Long")

# 6. 浅灰色 大十字 (对称)
# 消耗: 0, 时间: 3, 收入: 1
encoder.add(0, 3, 1, """
.#..
####
.#..
""", "Grey_Big_Cross")

# 16-21 图4 
encoder.add(10, 5, 3, """
####
##..
""", "Red_Hook_5")

encoder.add(8, 6, 3, """
##。
##.
.##
""", "Blue_Steps")

encoder.add(5, 3, 1, """
.##.
####
.##.
""", "Red_Complex_Cross")

encoder.add(4, 2, 0, """
#.
##
##
.#
""", "Ginger_Steps")

encoder.add(3, 6, 2, """
.##
##.
.##
""", "Blue_Wave_H")

encoder.add(7, 1, 1, """
#####
""", "Slim_1x5")

# 22-27图5

encoder.add(4, 6, 2, """
##
.#
.#
""", "Brown_L_Offset")

encoder.add(3, 3, 1, "####", "Yellow_1x4")

encoder.add(5, 4, 2, """
.#.
###
.#.
""", "Pink_Small_Cross")

encoder.add(7, 6, 3, """
.#
##
#.
""", "Red_L_Stair")

encoder.add(2, 1, 0, """
.#.
.##
##.
.#.
""", "Red_L_Stair")

encoder.add(4, 2, 1, """
##
#.
#.
""", "Red_L_Stair")

# 38-33
# 1. 左上：浅绿色 Z 型 (Cost: 2, Time: 3, Income: 1)
encoder.add(2, 3, 1, """
.###
##..
""", "Light_Green_Z")

# 2. 中上：蓝色阶梯型 (Cost: 10, Time: 4, Income: 3)
encoder.add(10, 4, 3, """
..#
.##
##.
""", "Blue_Stairs")

# 3. 右上：橙色 L 型 (Cost: 2, Time: 2, Income: 0)
encoder.add(2, 2, 0, """
.#
##
##
""", "Orange_L")

# 4. 左下：碎花 L 型 (Cost: 1, Time: 3, Income: 0)
encoder.add(1, 3, 0, """
##
.#
""", "Floral_L")

# 5. 中下：深绿色 T 型 (Cost: 5, Time: 5, Income: 2)
encoder.add(5, 5, 2, """
.#.
.#.
###
""", "Green_Cross")

# 6. 右下：青色长 Z 型 (Cost: 1, Time: 2, Income: 0)
encoder.add(1, 2, 0, """
...#
####
#...
""", "Cyan_Long_Z")
# ... (其他 add 调用)

# 确保 33 个块都填进去了
with open("patch_data.h", "w", encoding="utf-8") as f:
    f.write(encoder.generate_cpp_header())

print(f"成功！已生成 {len(encoder.patches)} 个拼块数据到 patch_data.h")