#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

const int CANNY_THRESHOLD = 25;
const int MEDIAN_BLUR_K_SIZE = 9;
const int MORPH_K_SIZE = 1;

cv::RNG rng;

std::vector<std::vector<cv::Point>> filter_contour(const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<std::vector<cv::Point>> contours_filtered;
    for (size_t i = 0; i < contours.size(); ++i) {
        try {
            std::vector<cv::Point> convex_hull;
            cv::convexHull(contours[i], convex_hull);
            double area_hull = cv::contourArea(convex_hull);

            if (600 < area_hull) {
                double circumference_hull = cv::arcLength(convex_hull, true);
                double circularity_hull = (4 * CV_PI * area_hull) / (circumference_hull * circumference_hull);

                if (0.8 < circularity_hull) {
                    contours_filtered.push_back(convex_hull);
                }
            }
        }
        catch (cv::Exception& e) {
            std::cerr << "Exception: " << e.what() << " for contour " << i << std::endl;
        }
    }
    return contours_filtered;
}

cv::Mat draw_ellipse(cv::Mat& _drawing, const std::vector<std::vector<cv::Point>>& _contours_filtered) {
    std::vector<cv::RotatedRect> minEllipse(_contours_filtered.size());
    for (size_t i = 0; i < _contours_filtered.size(); ++i) {
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        minEllipse[i] = cv::fitEllipse(_contours_filtered[i]);
        cv::drawContours(_drawing, _contours_filtered, static_cast<int>(i), color);
        cv::ellipse(_drawing, minEllipse[i], color, 2);
    }
    return _drawing;
}

int main() {
    cv::VideoCapture pcap("libcamerasrc ! video/x-raw,width=640,height=480 ! videoflip method=clockwise ! videoconvert ! appsink drop=True");
    if (!pcap.isOpened()) {
        std::cout << "Pi camera is not working:" << std::endl;
        return -1;
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(MORPH_K_SIZE, MORPH_K_SIZE));

    // Timing Execution
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (true) {
        cv::Mat pframe;
        bool pret = pcap.read(pframe);

        if (pret) {
            pframe = pframe(cv::Rect(0, 0, 480, 480));
            cv::Mat output = pframe.clone();
            cv::cvtColor(output, output, cv::COLOR_BGR2GRAY);
            cv::medianBlur(output, output, MEDIAN_BLUR_K_SIZE);
            cv::morphologyEx(output, output, cv::MORPH_OPEN, kernel);
            cv::Mat canny;
            cv::Canny(output, canny, CANNY_THRESHOLD, CANNY_THRESHOLD * 2);

            // Profiling
            // Uncomment the following lines for profiling
            // auto start_profiling = std::chrono::high_resolution_clock::now();

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(canny, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            std::vector<std::vector<cv::Point>> contours_filtered = filter_contour(contours);
            cv::Mat drawing = cv::Mat::zeros(canny.size(), CV_8UC3);
            drawing = draw_ellipse(output, contours_filtered);

            // Profiling
            // Uncomment the following lines for profiling
            // auto end_profiling = std::chrono::high_resolution_clock::now();
            // auto profiling_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_profiling - start_profiling).count();
            // std::cout << "Profiling Time: " << profiling_time << " milliseconds" << std::endl;

            cv::imshow("pupil", drawing);
            frame_count++;
            if (frame_count==1) {
                break;
            }
        }

        if (cv::waitKey(1) & 0xFF == 'q') {
            break;
        }
    }

    // Timing Execution
    auto end_time = std::chrono::high_resolution_clock::now();
    auto execution_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "Total Frames: " << frame_count << std::endl;
    std::cout << "Total Execution Time: " << execution_time << " seconds" << std::endl;
    std::cout << "Average FPS: " << static_cast<double>(frame_count) / execution_time << std::endl;

    pcap.release();
    cv::destroyAllWindows();

    return 0;
}



// #include <iostream>
// #include <chrono>

// void pupil_detection_algorithm(/* image parameters here */) {
//     // Your C++ implementation of the pupil detection algorithm here
// }

// int main() {
//     // Load the LPW dataset or use your own dataset
//     // std::vector<Image> images = load_lpw_dataset();

//     // Timing Execution
//     auto start_time = std::chrono::high_resolution_clock::now();
//     for (const auto& image : images) {
//         pupil_detection_algorithm(/* pass image parameters */);
//     }
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//     std::cout << "C++ Execution Time: " << execution_time << " milliseconds" << std::endl;

//     // Profiling Tools
//     // Uncomment the following lines for profiling
//     // std::cout << "Profiling..." << std::endl;
//     // auto start_profiling = std::chrono::high_resolution_clock::now();
//     // for (const auto& image : images) {
//     //     pupil_detection_algorithm(/* pass image parameters */);
//     // }
//     // auto end_profiling = std::chrono::high_resolution_clock::now();
//     // auto profiling_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_profiling - start_profiling).count();
//     // std::cout << "Profiling Time: " << profiling_time << " milliseconds" << std::endl;

//     // Benchmarking
//     // You can add more detailed benchmarking as needed
//     // auto benchmark_start = std::chrono::high_resolution_clock::now();
//     // for (int i = 0; i < NUM_ITERATIONS; ++i) {
//     //     pupil_detection_algorithm(/* pass image parameters */);
//     // }
//     // auto benchmark_end = std::chrono::high_resolution_clock::now();
//     // auto benchmark_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(benchmark_end - benchmark_start).count();
//     // std::cout << "Benchmark Execution Time: " << benchmark_execution_time << " milliseconds" << std::endl;

//     // Resource Usage
//     // You can use system commands or external libraries to gather resource usage data

//     return 0;
// }
