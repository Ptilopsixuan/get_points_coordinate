from PIL import Image, ImageDraw, ImageEnhance
from collections import defaultdict
import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


def detect_plot_frame(image_path, axis_color_threshold=250, axis_width=0):
    """
    精确版四边检测 - 生成紧凑绘图区域并保留四边颜色标识
    
    参数:
        image_path: 图片路径
        axis_color_threshold: 边框颜色阈值(0-255)
        axis_width: 边框线预估宽度(像素)
        
    返回:
        {
            'plot_area': (x1, y1, x2, y2),  # 紧凑绘图区域
            'borders': {  # 检测到的四边信息
                'top': (y_pos, x_start, x_end),
                'bottom': (y_pos, x_start, x_end),
                'left': (x_pos, y_start, y_end),
                'right': (x_pos, y_start, y_end)
            },
            'debug_img': Image  # 带标注的图像
        }
    """
    # 1. 图像加载和预处理
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    # 增强对比度（可选）
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    gray_img = img.convert("L")
    pixels = gray_img.load()
    
    # 2. 四边检测函数，只能检测到边界在整个图像最外围1/4区域的情况
    def detect_border(orientation):
        """检测指定方向的边框"""
        border_candidates = defaultdict(int)
        scan_range = None
        main_dim = None
        
        if orientation == 'top':
            scan_range = range(0, height//4)  # 只扫描上部1/4区域
            main_dim = width
        elif orientation == 'bottom':
            scan_range = range(height*3//4, height)  # 只扫描下部1/4区域
            main_dim = width
        elif orientation == 'left':
            scan_range = range(0, width//4)  # 只扫描左部1/4区域
            main_dim = height
        elif orientation == 'right':
            scan_range = range(width*3//4, width)  # 只扫描右部1/4区域
            main_dim = height
        
        for pos in scan_range:
            dark_pixels = 0
            for i in range(main_dim):
                if orientation in ['top', 'bottom']:
                    x, y = i, pos
                else:
                    x, y = pos, i
                
                if pixels[x, y] < axis_color_threshold:
                    dark_pixels += 1
            
            if dark_pixels > main_dim * 0.7:  # 70%像素符合条件
                border_candidates[pos] = dark_pixels
        
        if not border_candidates:
            return None
        
        # 选择最符合条件的边界位置
        if orientation in ['top', 'left']:
            border_pos = max(border_candidates.items(), key=lambda x: x[0])[0]
        else:
            border_pos = max(border_candidates.items(), key=lambda x: x[1])[0]
        
        return border_pos
        
    # 3. 执行四边检测
    borders = {
        'top': detect_border('top'),
        'bottom': detect_border('bottom'),
        'left': detect_border('left'),
        'right': detect_border('right')
    }

    debug_img = img.copy()
    draw = ImageDraw.Draw(debug_img)
    
    for side in borders.keys():
        if side in ['top', 'bottom']:
            pos, start, end = borders[side], 0, width - 1
            draw.line([(start, pos), (end, pos)], fill = (255, 255, 0), width = 1)
        else:
            pos, start, end = borders[side], 0, height - 1
            draw.line([(pos, start), (pos, end)], fill = (255, 255, 0), width = 1)
    
    os.makedirs(".\\result\\coordinate\\", exist_ok=True)
    debug_img.save(f".\\result\\coordinate\\{image_path.split("\\")[-1].split(".")[0]}.png")

    return borders

def get_points(image_path, relations = ["<", "<", "<"], thresholds=[255, 255, 255], min_distance=25, visualize=True, border = None):
    """
    获取所有色点坐标并合并邻近点
    
    参数:
        image_path: 图像路径
        relation: 图像上点的颜色与目标颜色关系，推荐顺序rgb
        threshold: 红色通道最小值(0-255)，顺序推荐rgb
        min_distance: 合并点的最小距离(像素)
        visualize: 是否生成可视化结果
        
    返回:
        {
            'merged_points': [(x1,y1), ...],  # 合并后的点
            'merged_image': Image  # 标注图像(如果visualize=True)
        }
    """
    # 0. 参数检查
    op = ["<", "<=", "==", ">=", ">", "!="]
    for r in relations:
        if r not in op:
            relations[relations.index(r)] = "<="
            thresholds[relations.index(r)] = 255

    for t in thresholds:
        if not isinstance(t, int) or t < 0 or t > 255:
            print("参数错误: threshold must be a int in range(0, 255)")
            thresholds[thresholds.index(t)] = 255

    # 1. 加载图像
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pixels = np.array(img)
    
    # 获取所有点坐标
    # points = []
    # for y in range(border["top"], border["bottom"] + 1): # 在图像边界内遍历，从上向下遍历
    #     for x in range(border["left"], border["right"] + 1):
    #         r, g, b = pixels[y, x]

    #         condition = eval(f"{r}{relation[0]}{threshold[0]} and {g}{relation[1]}{threshold[1]} and {b}{relation[2]}{threshold[2]}") 太慢了
            
    #         if condition:
    #             points.append([x, height - y])  # 转换为数学坐标系

    # 提取 ROI（Region of Interest）区域
    roi = pixels[border["top"] : border["bottom"]+1, border["left"] : border["right"]+1]

    # 分解 RGB 通道
    r_channel = roi[:, :, 0]
    g_channel = roi[:, :, 1]
    b_channel = roi[:, :, 2]

    # 创建运算符映射字典（支持 >, <, >=, <=, ==）
    op_map = {
        '>': np.greater, 
        '<': np.less, 
        '>=': np.greater_equal, 
        '<=': np.less_equal, 
        '==': np.equal
        }

    # 生成各通道的布尔掩码
    mask_r = op_map[relations[0]](r_channel, thresholds[0])
    mask_g = op_map[relations[1]](g_channel, thresholds[1])
    mask_b = op_map[relations[2]](b_channel, thresholds[2])

    # 组合最终掩码（同时满足三个条件）
    final_mask = mask_r & mask_g & mask_b

    # 获取满足条件的坐标（相对 ROI 的坐标）
    y_coords, x_coords = np.where(final_mask)

    # 转换为原始图像坐标系（注意坐标变换）
    points = [
        [x + border["left"], (border["top"] + y)]  # 原始坐标计算
        for x, y in zip(x_coords, y_coords)
    ]

    # 如果需要转换为数学坐标系（原点在左下角）
    height = pixels.shape[0]
    points = [[x, height - y] for x, y in points]
    
    if not points:
        return {'error': 'No points detected'}
    

    # 2. 将点转换为numpy数组
    points_array = np.array(points)
    
    # 3. 使用DBSCAN聚类合并邻近点
    clustering = DBSCAN(eps=min_distance, min_samples=1).fit(points_array)
    labels = clustering.labels_
    
    # 4. 计算每个簇的平均位置
    merged_points = []
    for label in set(labels):
        cluster_points = points_array[labels == label]
        merged_points.append(tuple(np.mean(cluster_points, axis=0).astype(int)))
    
    # 5. 可视化结果
    result = {'merged_points': merged_points}
    
    if visualize:
        # 创建标注图像
        marked_img = img.copy()
        draw = ImageDraw.Draw(marked_img)
        
        # 绘制合并点(绿色圆圈)
        for x, y_img in merged_points:
            y = height - y_img
            draw.ellipse([(x-5, y-5), (x+5, y+5)], outline='green', width=2)
        
        result['merged_image'] = marked_img

    return result

def detect(path=".\\original", type=".png", colors=None, min_distance=25, visualize=True):
    """
    检测颜色并将数值导入Excel表格
    :param path: 图片路径 default = ".\\original"
    :param type: 图片类型 default = ".png"
    :param colors: 颜色字典，如 {
        "red": [">", "<", "<", 250, 255, 10],
        "blue": ["<", "<", ">", 10, 255, 250],
        ...
    }
    """
    # 初始化颜色数据存储结构
    color_data = {color: {"x": [], "y": []} for color in colors} if colors else {}
    files = os.listdir(path)
    
    for file in files:
        if not file.endswith(type):
            continue
        
        print(file)

        # 基础数据初始化
        base_data = {
            "name": [file.split(".")[0]],
            "o_x": [], "o_y": [], 
            "u_x": [], "u_y": []
        }
        
        # 处理图像边界
        full_path = os.path.join(path, file)
        borders = detect_plot_frame(full_path)
        
        # 填充基础坐标
        base_data["o_x"].append(borders["left"])
        base_data["o_y"].append(borders["bottom"])
        base_data["u_x"].append(borders["right"])
        base_data["u_y"].append(borders["top"])
        
        # 处理每个颜色通道
        for color, params in colors.items():
            # 执行颜色检测
            result = get_points(
                full_path,
                relations=params[:3],
                thresholds=params[3:],
                border=borders,
                min_distance = min_distance,
                visualize = visualize
                )
            
            print(f"{color}: {len(result["merged_points"])}")
            
            # 保存检测结果图片
            save_dir = f".\\result\\{color}_coor"
            os.makedirs(save_dir, exist_ok=True)
            result['merged_image'].save(f"{save_dir}\\{file.split('.')[0]}.png")
            
            # 收集坐标点
            x_coords = []
            y_coords = []
            for _, (x, y) in enumerate(result['merged_points'], 1):
                x_coords.append(x)
                y_coords.append(y)
            
            # 存储颜色数据
            color_data[color]["x"] = x_coords
            color_data[color]["y"] = y_coords

        # 计算最大长度
        max_len = max(
            len(base_data["name"]),
            *[len(data["x"]) for data in color_data.values()]
        )
        
        # 数据对齐填充
        for key in base_data:
            if key == "name":
                base_data[key] += [0] * (max_len - len(base_data[key]))
            else:
                base_data[key] = [base_data[key][0]] * max_len if base_data[key] else []
        
        for color in colors:
            for coord in ["x", "y"]:
                data_len = len(color_data[color][coord])
                color_data[color][coord] += [0] * (max_len - data_len)
        
        # 构建DataFrame
        df_data = {
            "Name": base_data["name"],
            "O_X": base_data["o_x"],
            "O_Y": base_data["o_y"],
            "U_X": base_data["u_x"],
            "U_Y": base_data["u_y"]
        }
        
        # 添加颜色数据列
        for color in colors:
            df_data[f"{color.upper()}_X"] = color_data[color]["x"]
            df_data[f"{color.upper()}_Y"] = color_data[color]["y"]
        
        df = pd.DataFrame(df_data)
        
        # 导出到Excel
        with pd.ExcelWriter(".\\result\\result.xlsx", engine="openpyxl", 
                         mode="a" if os.path.exists(".\\result.xlsx") else "w",
                         if_sheet_exists="replace" if os.path.exists(".\\result.xlsx") else None,
                         ) as writer:
            df.to_excel(
                writer,
                sheet_name=base_data["name"][0][:31],  # Excel表名最长31字符
                index=False
            )
