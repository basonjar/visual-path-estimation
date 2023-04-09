#include <array>
#include <cstdlib>
#include <filesystem>
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
#include <queue>
#include <ranges>
#include <vector>

auto K = std::make_shared<gtsam::Cal3_S2>(960, 540, 0, 1344, 1344);

namespace fs = std::filesystem;

struct Image {
    cv::Mat mat;
    std::vector<cv::KeyPoint> features;
    cv::Mat descriptors;
};

struct FeatureComponent {
    size_t parentImage, parentFeature;

    auto operator<=>(FeatureComponent const&) const = default;
};

constexpr size_t FEATURES_PER_IMAGE = 500;

using FeatureGraph = std::vector<std::array<FeatureComponent, FEATURES_PER_IMAGE>>;

[[nodiscard]] FeatureComponent findParent(FeatureGraph const& graph, FeatureComponent const& component) {
    FeatureComponent const& parent = graph[component.parentImage][component.parentFeature];
    return parent == component ? parent : findParent(graph, parent);
}

void unionComponents(FeatureGraph& graph, FeatureComponent const& a, FeatureComponent const& b) {
    FeatureComponent const& parentA = findParent(graph, a);
    FeatureComponent const& parentB = findParent(graph, b);

    if (parentA == parentB) return;

    graph[parentA.parentImage][parentA.parentFeature] = parentB;
}

auto loadImages(fs::path const& imageDirectoryPath) {
    std::cout << "Loading images from " << imageDirectoryPath.stem() << std::endl;
    std::vector<Image> images;
    size_t i = 0;
    for (fs::directory_entry const& imagePath: fs::directory_iterator(imageDirectoryPath)) {
        if (++i == 4) break;

        std::cout << "\tLoading " << imagePath.path() << std::endl;

        cv::Mat mat = cv::imread(imagePath.path(), cv::IMREAD_COLOR);
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> features;

        cv::Ptr<cv::ORB> orb = cv::ORB::create();

        orb->detectAndCompute(mat, cv::noArray(), features, descriptors);

        images.emplace_back(Image{mat, features, descriptors});
    }
    return images;
}

int main() {
    auto imageDirectoryPath = fs::current_path() / "data" / "caterpillar";

    // Load images and calculate features
    std::vector<Image> images = loadImages(imageDirectoryPath);

    // TODO: optimize by vectorizing the distance matrix calculation
    //    // Compute L2 distance between N (500 x 32) matrices
    //    std::vector<std::array<std::array<double, 32>, FEATURES_PER_IMAGE>> distances((images.size() * images.size() - 1) / 2);
    //    for (int i = 0; i < images.size(); ++i) {
    //        for (int j = i + 1; j < images.size(); ++j) {
    //        }
    //    }

    // Create graph of connections between features across multiple images
    FeatureGraph featureGraph;
    featureGraph.resize(images.size());// N x 500 matrix to be used for connected components

    for (size_t i = 0; i < images.size(); ++i) {
        for (size_t f = 0; f < FEATURES_PER_IMAGE; ++f) {
            featureGraph[i][f] = FeatureComponent{i, f};
        }
    }

    for (size_t img1 = 0; img1 < images.size(); ++img1) {
        for (size_t f1 = 0; f1 < FEATURES_PER_IMAGE; ++f1) {

            for (size_t img2 = 0; img2 < images.size(); ++img2) {
                if (img1 == img2) continue;

                using p_t = std::pair<double, size_t>;
                std::priority_queue<p_t, std::vector<p_t>, std::greater<>> distances;

                for (size_t f2 = 0; f2 < FEATURES_PER_IMAGE; ++f2) {
                    distances.emplace(cv::norm(images[img1].descriptors.row(f1), images[img2].descriptors.row(f2)), f2);
                }

                auto [d1, f2] = distances.top();
                distances.pop();
                auto [d2, _] = distances.top();
                distances.pop();

                double ratio = d1 / d2;

                if (ratio < 0.8) {
                    std::printf("Found match between image %zu feature %zu and image %zu feature %zu\n", img1, f1, img2, f2);
                    unionComponents(featureGraph, FeatureComponent{img1, f1}, FeatureComponent{img2, f2});
                }
            }
        }
    }

    std::set<FeatureComponent> uniqueComponents;
    for (size_t img = 0; img < images.size(); ++img) {
        for (size_t f = 0; f < FEATURES_PER_IMAGE; ++f) {
            uniqueComponents.insert(findParent(featureGraph, FeatureComponent{img, f}));
        }
    }
    std::cout << "Unique components: " << uniqueComponents.size() << std::endl;

    //    std::vector<cv::Mat> uniqueDescriptors;
    //
    //    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    //
    //    std::vector<cv::DMatch> matches1;
    //    matcher->match(images[1].descriptors, images[0].descriptors, matches1);
    //
    //    std::vector<cv::DMatch> matches2;
    //    matcher->match(images[2].descriptors, images[0].descriptors, matches2);

    //    gtsam::NonlinearFactorGraph graph;
    //
    //    graph.addPrior(gtsam::Symbol('x', 0), gtsam::Pose3(), gtsam::noiseModel::Isotropic::Sigma(6, 0.1));
    //
    //    for (size_t j = 0; j < images.size(); ++j) {
    //        Image const& image = images[j];
    //        for (size_t i = 0; i < image.features.size(); ++i) {
    //            auto const& descriptor = image.descriptors.row(static_cast<int>(i));
    //
    //            auto it = std::ranges::find_if(uniqueDescriptors, [&](cv::Mat const& existingDescriptor) {
    //                return cv::norm(existingDescriptor, descriptor) < 500.0;
    //            });
    //            if (it == uniqueDescriptors.end()) continue;
    //
    //            graph.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(
    //                    gtsam::Point2(image.features[i].pt.x, image.features[i].pt.y),
    //                    gtsam::noiseModel::Isotropic::Sigma(2, 1.0),
    //                    gtsam::Symbol('x', j),
    //                    gtsam::Symbol('l', std::distance(uniqueDescriptors.begin(), it)),
    //                    K);
    //        }
    //    }
    //
    //    graph.addPrior(gtsam::Symbol('l', 0), gtsam::Point3(), gtsam::noiseModel::Isotropic::Sigma(3, 0.1));
    //
    //    gtsam::Values initial;
    //    for (size_t i = 0; i < images.size(); ++i) {
    //        initial.insert(gtsam::Symbol('x', i), gtsam::Pose3());
    //    }
    //    for (size_t i = 0; i < uniqueDescriptors.size(); ++i) {
    //        initial.insert(gtsam::Symbol('l', i), gtsam::Point3());
    //    }
    //
    //    gtsam::Values result = gtsam::DoglegOptimizer(graph, initial).optimize();
    //    result.print("result: ");

    // TODO: Average descriptor values
    // TODO: Implement better matching

    return EXIT_SUCCESS;
}
