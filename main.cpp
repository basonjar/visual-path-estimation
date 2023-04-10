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
#include <opencv2/imgproc.hpp>
#include <queue>
#include <ranges>
#include <vector>

auto K = std::make_shared<gtsam::Cal3_S2>(960, 540, 0, 1344, 1344);

namespace fs = std::filesystem;

struct Image {
    cv::Mat mat;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct FeatureComponent {
    size_t rootImage, rootFeature;
    bool isReferenced;

    auto operator<=>(FeatureComponent const&) const = default;
};

using FeatureGraph = std::vector<std::vector<FeatureComponent>>;

[[nodiscard]] FeatureComponent& findRoot(FeatureGraph& graph, FeatureComponent const& component) {
    FeatureComponent& root = graph[component.rootImage][component.rootFeature];
    return root == component ? root : findRoot(graph, root);
}

void unionComponents(FeatureGraph& graph, FeatureComponent const& a, FeatureComponent const& b) {
  FeatureComponent& rootA = findRoot(graph, a);
  FeatureComponent& rootB = findRoot(graph, b);

    if (rootA == rootB) return;

    rootA.isReferenced = rootB.isReferenced = true;

    graph[rootA.rootImage][rootA.rootFeature] = rootB;
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

        cv::Ptr<cv::BRISK> descriptorExtractor = cv::BRISK::create();

        descriptorExtractor->detectAndCompute(mat, cv::noArray(), features, descriptors);

        images.emplace_back(Image{mat, features, descriptors});
    }
    return images;
}

int main() {
    auto imageDirectoryPath = fs::current_path() / "data" / "caterpillar";

    // Load images and calculate features
    std::vector<Image> images = loadImages(imageDirectoryPath);

    // Create graph of connections between features across multiple images
    FeatureGraph featureGraph;
    featureGraph.resize(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        featureGraph[i].resize(images[i].keypoints.size());
        for (size_t f = 0; f < images[i].keypoints.size(); ++f) {
            featureGraph[i][f] = FeatureComponent{i, f, false};
        }
    }

    for (size_t img1 = 0; img1 < images.size(); ++img1) {
        for (size_t img2 = 0; img2 < images.size(); ++img2) {
            if (img1 == img2) continue;

            // TODO: try out cross check, alt. to rati otest
            auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
            using BestMatches = std::vector<cv::DMatch>;
            std::vector<BestMatches> matches;
            matcher->knnMatch(images[img1].descriptors, images[img2].descriptors, matches, 2);

            BestMatches goodMatches;
            for (BestMatches const& bestMatch: matches) {
                assert(bestMatch.size() == 2);

                auto const& [firstBest, secondBest] = std::tuple{bestMatch[0], bestMatch[1]};
                double ratio = firstBest.distance / secondBest.distance;

                if (ratio < 0.2) {
                    std::printf("Match: %d -> %d\n", firstBest.queryIdx, firstBest.trainIdx);
                    unionComponents(featureGraph, FeatureComponent(img1, firstBest.queryIdx), FeatureComponent(img2, firstBest.trainIdx));
                    goodMatches.push_back(firstBest);
                }
            }

            //            cv::Mat m;
            //            cv::drawMatches(images[img1].mat, images[img1].keypoints,
            //                            images[img2].mat, images[img2].keypoints,
            //                            goodMatches, m);
            //            cv::resize(m, m, {}, 0.45, 0.45);
            //            cv::imshow("Matches", m);
            //            cv::waitKey(0);
        }
    }

    std::set<FeatureComponent> uniqueComponents;
    for (size_t i = 0; i < featureGraph.size(); ++i) {
        for (size_t f = 0; f < featureGraph[i].size(); ++f) {
            if (!featureGraph[i][f].isReferenced) continue;

            uniqueComponents.insert(findRoot(featureGraph, FeatureComponent{i, f}));
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
