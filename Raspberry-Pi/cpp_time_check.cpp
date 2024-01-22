#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <sys/resource.h>
#include <fstream>
#include <string>

const int CANNY_THRESHOLD = 25;
const int MEDIAN_BLUR_K_SIZE = 9;
const int MORPH_K_SIZE = 1;
double blur_final_time = 0;


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

cv::Mat process_frame(cv::Mat& pframe) {
    cv::cvtColor(pframe, pframe, cv::COLOR_BGR2GRAY);
    
    auto blur_start_time = std::chrono::high_resolution_clock::now();
    cv::medianBlur(pframe, pframe, MEDIAN_BLUR_K_SIZE);
    auto blur_end_time = std::chrono::high_resolution_clock::now();
    auto blur_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(blur_end_time - blur_start_time).count();
    blur_final_time += blur_execution_time/1000.00;
    
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(MORPH_K_SIZE, MORPH_K_SIZE));
    cv::morphologyEx(pframe, pframe, cv::MORPH_OPEN, kernel);
    cv::Mat canny;
    cv::Canny(pframe, canny, CANNY_THRESHOLD, CANNY_THRESHOLD * 2);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point>> contours_filtered = filter_contour(contours);
    cv::Mat drawing = cv::Mat::zeros(canny.size(), CV_8UC3);
    drawing = draw_ellipse(pframe, contours_filtered);
    return drawing;
}

int main() {
    //cv::VideoCapture pcap("libcamerasrc ! video/x-raw,width=640,height=480 ! videoflip method=clockwise ! videoconvert ! appsink drop=True");
    
    //reading from the file
    std::string path = "/home/demo/Desktop/LPW data/LPW/2/4.avi";
    cv::VideoCapture pcap(path);
    
    if (!pcap.isOpened()) {
        std::cout << "Pi camera is not working" << std::endl;
        return -1;
    }

    

    // Timing Execution
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    int ncalls = 100;
    double process_final_time = 0;
    

    while (true) {
        cv::Mat pframe;
        bool pret = pcap.read(pframe);

        if (pret) {
            pframe = pframe(cv::Rect(0, 0, 480, 480));
            
            
            //cv::Mat output_frame =  process_frame(pframe);
            //profiling
            auto process_start_time = std::chrono::high_resolution_clock::now();
            
            
            cv::imshow("pupil", process_frame(pframe));
            
            //profiling
            auto process_end_time = std::chrono::high_resolution_clock::now();
            auto process_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(process_end_time - process_start_time).count();
            process_final_time += process_execution_time/1000.00;
            
            
            frame_count++;
            if (frame_count==ncalls) {
                break;
            }
        }

        if (cv::waitKey(1) & 0xFF == 'q') {
            break;
        }
    }

    // Timing Execution
    auto end_time = std::chrono::high_resolution_clock::now();
    auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Total Frames: " << frame_count << std::endl;
    double final_time = execution_time/1000.00;
    std::cout << "Total Execution Time: " << final_time << " seconds" << std::endl;
    std::cout << "Average FPS: " << static_cast<double>(frame_count) / final_time << std::endl;
    std::cout << "Cumulative process_frame() time: " << process_final_time << " -  Per call avg: "<< process_final_time/ncalls << std::endl;
    std::cout << "Cumulative Median Blur time: " << blur_final_time << " -  Per call avg: "<< blur_final_time/ncalls << std::endl;
    

    pcap.release();
    cv::destroyAllWindows();

    return 0;
}



