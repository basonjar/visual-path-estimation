#include <cstdlib>
#include <filesystem>
#include <format>
#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <ranges>
#include <vector>

auto K = std::make_shared<gtsam::Cal3_S2>(960, 540, 0, 1344, 1344);

namespace fs = std::filesystem;

struct Image {
    cv::Mat mat;
    std::vector<cv::KeyPoint> features;
    cv::Mat descriptors;
};

int main() {
    auto imageDirectoryPath = fs::current_path() / "data" / "caterpillar";

    std::cout << "Loading images from " << imageDirectoryPath.stem() << std::endl;

    std::vector<Image> images;
    for (fs::directory_entry const& imagePath:
         fs::directory_iterator(imageDirectoryPath) |
                 std::views::filter([](fs::directory_entry const& path) {
                     return path.path().extension() == ".jpg";
                 }) |
                 std::views::take(10)) {

        std::cout << "\tLoading " << imagePath.path() << std::endl;

        cv::Mat mat = cv::imread(imagePath.path(), cv::IMREAD_COLOR);
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> features;

        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        orb->detectAndCompute(mat, cv::noArray(), features, descriptors);

        images.emplace_back(Image{mat, features, descriptors});

        //    cv::drawKeypoints(mat, features, mat, cv::Scalar(0, 255, 0),
        //    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        //
        //    cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
        //    cv::imshow("image", mat);
        //    cv::waitKey();
    }

    std::vector<cv::Mat> uniqueDescriptors;

    for (Image const& image: images) {
        std::cout << "Processing image for unique descriptors" << std::endl;

        for (size_t i = 0; i < image.features.size(); ++i) {
            auto const& descriptor = image.descriptors.row(static_cast<int>(i));

            bool matchesExisting = std::ranges::any_of(uniqueDescriptors, [&](cv::Mat const& existingDescriptor) {
                double norm = cv::norm(existingDescriptor, descriptor);
                return norm < 500.0;
            });
            if (matchesExisting) continue;

            uniqueDescriptors.push_back(descriptor);
        }
    }

    gtsam::NonlinearFactorGraph graph;

    graph.addPrior(gtsam::Symbol('x', 0), gtsam::Pose3(), gtsam::noiseModel::Isotropic::Sigma(6, 0.1));

    for (size_t j = 0; j < images.size(); ++j) {
        Image const& image = images[j];
        for (size_t i = 0; i < image.features.size(); ++i) {
            auto const& descriptor = image.descriptors.row(static_cast<int>(i));

            auto it = std::ranges::find_if(uniqueDescriptors, [&](cv::Mat const& existingDescriptor) {
                return cv::norm(existingDescriptor, descriptor) < 500.0;
            });
            if (it == uniqueDescriptors.end()) continue;

            graph.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(
                    gtsam::Point2(image.features[i].pt.x, image.features[i].pt.y),
                    gtsam::noiseModel::Isotropic::Sigma(2, 1.0),
                    gtsam::Symbol('x', j),
                    gtsam::Symbol('l', std::distance(uniqueDescriptors.begin(), it)),
                    K);
        }
    }

    graph.addPrior(gtsam::Symbol('l', 0), gtsam::Point3(), gtsam::noiseModel::Isotropic::Sigma(3, 0.1));

    gtsam::Values initial;
    for (size_t i = 0; i < images.size(); ++i) {
        initial.insert(gtsam::Symbol('x', i), gtsam::Pose3());
    }
    for (size_t i = 0; i < uniqueDescriptors.size(); ++i) {
        initial.insert(gtsam::Symbol('l', i), gtsam::Point3());
    }

    gtsam::Values result = gtsam::DoglegOptimizer(graph, initial).optimize();
    result.print("result: ");

    // TODO: Average descriptor values
    // TODO: Implement better matching

    return EXIT_SUCCESS;
}
