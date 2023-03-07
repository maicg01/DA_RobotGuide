# import pyrealsense2 as rs
# import numpy as np

# depth_scale = 0.001
# # Khởi tạo RealSense pipeline
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# pipeline.start(config)

# # Lấy frame mới nhất từ camera
# frames = pipeline.wait_for_frames()
# depth_frame = frames.get_depth_frame()

# # Lấy thông tin về hình ảnh
# width = depth_frame.get_width()
# height = depth_frame.get_height()

# # Chuyển đổi frame depth sang ma trận numpy
# depth_image = np.asanyarray(depth_frame.get_data())

# # Tính toán khoảng cách tại điểm (x, y)
# x = 320 # giá trị x của điểm ảnh
# y = 240 # giá trị y của điểm ảnh
# depth_in_meters = depth_image[y, x] * depth_scale

# # In ra khoảng cách tại điểm ảnh
# print("Khoảng cách tại điểm ảnh ({}, {}) là {} mét".format(x, y, depth_in_meters))

# # Dừng pipeline và giải phóng tài nguyên
# pipeline.stop()


# import cv2
# import numpy as np
# import pyrealsense2 as rs

# # Khởi tạo camera Realsense
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# pipeline.start(config)

# try:
#     while True:
#         # Lấy khung hình từ camera
#         frames = pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()

#         # Lấy giá trị khoảng cách tại tọa độ (320, 240) (giữa khung hình)
#         distance = depth_frame.get_distance(320, 240)

#         # Hiển thị giá trị khoảng cách lên màn hình
#         img = np.zeros((512, 512, 3), dtype=np.uint8)
#         cv2.putText(img, 'Distance: {:.2f}m'.format(distance), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         cv2.imshow("Depth", img)

#         # Thoát khi nhấn phím ESC
#         key = cv2.waitKey(1)
#         if key == 27:
#             break

# finally:
#     pipeline.stop()
#     cv2.destroyAllWindows()


import cv2
import pyrealsense2 as rs
import numpy as np

# Khởi tạo kết nối với camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Khởi tạo bộ lọc cho ảnh độ sâu
depth_filter = rs.align(rs.stream.color)

while True:
    # Đọc ảnh từ camera
    frames = pipeline.wait_for_frames()
    aligned_frames = depth_filter.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # Lấy khoảng cách đến vật thể
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    dist_to_obj = depth_frame.get_distance(depth_intrin.width//2, depth_intrin.height//2)
    dist_to_obj = round(dist_to_obj, 2)

    # Hiển thị ảnh lấy từ camera và khoảng cách đến vật thể
    color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2BGRA)
    cv2.putText(color_image, 'Distance to object: {} m'.format(dist_to_obj), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Camera', color_image)

    # Đợi nhấn phím để thoát
    if cv2.waitKey(1) == ord('q'):
        break

# Dọn dẹp và thoát chương trình
pipeline.stop()
cv2.destroyAllWindows()