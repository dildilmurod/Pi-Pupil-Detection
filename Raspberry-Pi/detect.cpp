#include <opencv2/opencv.hpp>
#include <iostream>

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
    if (pcap.isOpened()) {
        std::cout << "Pi camera available:" << std::endl;
    }
    else{
        std::cout << "Pi camera is not working:" << std::endl;
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(MORPH_K_SIZE, MORPH_K_SIZE));

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
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(canny, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            std::vector<std::vector<cv::Point>> contours_filtered = filter_contour(contours);
            cv::Mat drawing = cv::Mat::zeros(canny.size(), CV_8UC3);
            drawing = draw_ellipse(output, contours_filtered);
            cv::imshow("pupil", drawing);
        }

        if (cv::waitKey(1) & 0xFF == 'q') {
            break;
        }
    }

    pcap.release();
    cv::destroyAllWindows();

    return 0;
}
