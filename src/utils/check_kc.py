import cv2
import pyrealsense2 as rs
import numpy as np


def check_kc(frames, depth_filter):
    aligned_frames = depth_filter.process(frames)
    depth_frame = aligned_frames.get_depth_frame()

    # Lấy khoảng cách đến vật thể
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    dist_to_obj = depth_frame.get_distance(depth_intrin.width//2, depth_intrin.height//2)
    dist_to_obj = round(dist_to_obj, 2)
    if dist_to_obj < 0.5:
        print("Khoang cach OK")
        return 1
    return 0

# # Khởi tạo kết nối với camera
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# pipeline.start(config)

# # Khởi tạo bộ lọc cho ảnh độ sâu
# depth_filter = rs.align(rs.stream.color)

# while True:
#     # Đọc ảnh từ camera
#     frames = pipeline.wait_for_frames()
#     if check_kc(frames) == 1:
#         print("OK")

#     # # Hiển thị ảnh lấy từ camera và khoảng cách đến vật thể
#     # color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2BGRA)
#     # cv2.putText(color_image, 'Distance to object: {} m'.format(dist_to_obj), (10, 30),
#     #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     # cv2.imshow('Camera', color_image)

#     # # Đợi nhấn phím để thoát
#     # if cv2.waitKey(1) == ord('q'):
#     #     break

# # Dọn dẹp và thoát chương trình
# pipeline.stop()
# cv2.destroyAllWindows()



