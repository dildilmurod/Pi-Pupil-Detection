import cv2
import numpy as np
import random as rng
import time
import cProfile
import psutil

'''Change the parameters to conduct real-time detection'''
CANNY_THRESHOLD = 25
MEDIAN_BLUR_K_SIZE = 9
MORPH_K_SIZE = 1

def filter_contour(_contours):
    _contours_filtered = []
    for i, c in enumerate(_contours):
        try:
            convex_hull = cv2.convexHull(c)
            area_hull = cv2.contourArea(convex_hull)
            if 600 < area_hull:  # filtering based on area
                circumference_hull = cv2.arcLength(convex_hull, True)
                circularity_hull = (4 * np.pi * area_hull) / circumference_hull ** 2
                if 0.8 < circularity_hull:  # filtering based on circularity
                    _contours_filtered.append(convex_hull)
        except ZeroDivisionError:
            print("Division by zero for contour {}".format(i))
    return _contours_filtered

def draw_ellipse(_drawing, _contours_filtered):
    minEllipse = [None] * len(_contours_filtered)
    for i, c in enumerate(_contours_filtered):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        minEllipse[i] = cv2.fitEllipse(c)
        cv2.drawContours(_drawing, _contours_filtered, i, color)
        cv2.ellipse(_drawing, minEllipse[i], color=color, thickness=2)
    return _drawing

def process_frame(frame):
    output = frame.copy()
    src_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.medianBlur(src_gray, MEDIAN_BLUR_K_SIZE)
    kernel = np.ones((MORPH_K_SIZE, MORPH_K_SIZE), np.uint8)
    opening = cv2.morphologyEx(src_gray, cv2.MORPH_OPEN, kernel)
    canny = cv2.Canny(opening, CANNY_THRESHOLD, CANNY_THRESHOLD * 2)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_filtered = filter_contour(contours)
    drawing = np.zeros((canny.shape[0], canny.shape[1], 3), dtype=np.uint8)
    drawing = draw_ellipse(opening, contours_filtered)
    return drawing

srcPiCam = 'libcamerasrc ! video/x-raw,width=640,height=480 ! videoflip method=clockwise ! videoconvert ! appsink drop=True'
pcap = cv2.VideoCapture(srcPiCam)

if pcap.isOpened():
    print(f'Pi Camera is available.')

frame_count = 0

# Timing Execution
start_time = time.time()

#profiling
profiler = cProfile.Profile()

#resource usage
process = psutil.Process()

while True:
    pret, pframe = pcap.read()
    if pret:
        pframe = pframe[0:480, 0:480]
        
        profiler.enable()
        
        output_frame = process_frame(pframe)
        cv2.imshow('Processed Frame', output_frame)
        
        profiler.disable()
        
        cpu_percent = psutil.cpu_percent()
        memory_percent = process.memory_percent()

        frame_count += 1
        if frame_count==100:
            break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Timing Execution
end_time = time.time()
execution_time = end_time - start_time
print(f"Total Frames: {frame_count}")
print(f"Total Execution Time: {execution_time} seconds")
print(f"Average FPS: {frame_count / execution_time}")

profiler.print_stats(sort="cumulative")

print(f"Average cpu usage {cpu_percent}")
print(f"Average memory usage {memory_percent}")

pcap.release()
cv2.destroyAllWindows()


# import time
# import cProfile

# def pupil_detection_algorithm(image):
#     # Your Python implementation of the pupil detection algorithm here
#     pass

# def main():
#     # Load the LPW dataset or use your own dataset
#     # images = load_lpw_dataset()

#     # Timing Execution
#     start_time = time.time()
#     for image in images:
#         pupil_detection_algorithm(image)
#     execution_time = time.time() - start_time
#     print(f"Python Execution Time: {execution_time} seconds")

#     # Profiling Tools
#     # Uncomment the following lines for profiling
#     # profiler = cProfile.Profile()
#     # profiler.enable()
#     # for image in images:
#     #     pupil_detection_algorithm(image)
#     # profiler.disable()
#     # profiler.print_stats(sort='cumulative')

#     # Benchmarking
#     # You can add more detailed benchmarking as needed
#     # benchmark_start = time.time()
#     # for i in range(NUM_ITERATIONS):
#     #     pupil_detection_algorithm(images[i % len(images)])
#     # benchmark_execution_time = time.time() - benchmark_start
#     # print(f"Benchmark Execution Time: {benchmark_execution_time} seconds")

#     # Resource Usage
#     # You can use resource monitoring tools or libraries to gather resource usage data

# if __name__ == "__main__":
#     main()
