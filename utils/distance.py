foc = 500.0        # 镜头焦距,单位为cm
real_hight_bicycle = 26.04      # 自行车高度，注意单位是英寸
real_hight_car = 59.08      # 轿车高度
real_hight_motorcycle = 47.24      # 摩托车高度
real_hight_bus = 125.98      # 公交车高度
real_hight_truck = 137.79   # 卡车高度

# 自定义函数，单目测距
def detect_distance_car(h):
    dis_inch = (real_hight_car * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    return dis_m

def detect_distance_bicycle(h):
    dis_inch = (real_hight_bicycle * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm / 100
    return dis_m

def detect_distance_motorcycle(h):
    dis_inch = (real_hight_motorcycle * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    return dis_m

def detect_distance_bus(h):
    dis_inch = (real_hight_bus * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    return dis_m

def detect_distance_truck(h):
    dis_inch = (real_hight_truck * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    return dis_m








