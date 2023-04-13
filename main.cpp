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
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <vector>

auto K = std::make_shared<gtsam::Cal3_S2>(960, 540, 0, 1344, 1344);

namespace fs = std::filesystem;

struct Image {
    cv::Mat mat;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

using img_idx_t = size_t;
using feature_idx_t = size_t;
using unique_feature_idx_t = size_t;

struct FeatureComponent {
    img_idx_t imageIndex;
    feature_idx_t featureIndex;

    auto operator<=>(FeatureComponent const&) const = default;
};

struct FeatureComponentHash {
    [[nodiscard]] size_t operator()(FeatureComponent const& component) const {
        return std::hash<img_idx_t>{}(component.imageIndex) ^ std::hash<feature_idx_t>{}(component.featureIndex);
    }
};

struct ImageWithUniqueFeature {
    FeatureComponent component;
    cv::KeyPoint keypoint;

    auto operator==(ImageWithUniqueFeature const& other) const {
        return component.imageIndex == other.component.imageIndex;
    }
};

struct ImageWithUniqueFeatureHash {
    [[nodiscard]] size_t operator()(ImageWithUniqueFeature const& image) const {
        return std::hash<img_idx_t>{}(image.component.imageIndex);
    }
};

using FeatureGraph = std::unordered_map<FeatureComponent, FeatureComponent, FeatureComponentHash>;

[[nodiscard]] FeatureComponent& findRoot(FeatureGraph& graph, FeatureComponent const& component) {
    auto it = graph.find(component);
    if (it == graph.end()) {
        auto it = graph.emplace(component, component);
        return it.first->second;
    }
    auto& [_, parent] = *it;
    return parent == component ? parent : findRoot(graph, parent);
}

void unionComponents(FeatureGraph& graph, FeatureComponent const& a, FeatureComponent const& b) {
    FeatureComponent& rootA = findRoot(graph, a);
    FeatureComponent& rootB = findRoot(graph, b);

    if (rootA == rootB) return;

    graph[rootA] = rootB;
}

auto loadImages(fs::path const& imageDirectoryPath) {
    std::cout << "Loading images from " << imageDirectoryPath.stem() << std::endl;
    std::vector<Image> images;
    size_t i = 0;

    std::vector<fs::directory_entry> imagePaths;
    for (auto const& entry: fs::directory_iterator(imageDirectoryPath)) {
        imagePaths.push_back(entry);
    }
    std::ranges::sort(imagePaths, [](fs::directory_entry const& a, fs::directory_entry const& b) {
        return a.path().stem().string() < b.path().stem().string();
    });

    for (fs::directory_entry const& imagePath: imagePaths) {
        if (i++ == 30) break;

        std::cout << "\tLoading " << imagePath.path() << std::endl;

        cv::Mat mat = cv::imread(imagePath.path(), cv::IMREAD_COLOR);
        cv::Mat descriptors;
        std::vector<cv::KeyPoint> features;

        cv::Ptr<cv::ORB> descriptorExtractor = cv::ORB::create();

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

    for (size_t img1 = 0; img1 < images.size(); ++img1) {
        for (size_t img2 = img1 + 1; img2 < images.size(); ++img2) {
            //            if (img1 == img2) continue;

            //            auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
            //            using BestMatches = std::vector<cv::DMatch>;
            //            std::vector<BestMatches> matches;
            //            matcher->knnMatch(images[img1].descriptors, images[img2].descriptors, matches, 1);
            //
            //            BestMatches goodMatches;
            //
            //            for (BestMatches const& bestMatch: matches | std::views::filter([](BestMatches const& bestMatch) { return !bestMatch.empty(); })) {
            //                assert(bestMatch.size() == 1);
            //
            //                goodMatches.push_back(bestMatch[0]);
            //                auto const& firstBest = bestMatch[0];
            //                unionComponents(featureGraph, FeatureComponent(img1, firstBest.queryIdx), FeatureComponent(img2, firstBest.trainIdx));
            //            }

            auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
            using BestMatches = std::vector<cv::DMatch>;
            std::vector<BestMatches> matches;
            matcher->knnMatch(images[img1].descriptors, images[img2].descriptors, matches, 2);

            BestMatches goodMatches;

            for (BestMatches const& bestMatch: matches) {
                assert(bestMatch.size() == 2);

                auto const& [firstBest, secondBest] = std::tuple{bestMatch[0], bestMatch[1]};
                double ratio = firstBest.distance / secondBest.distance;

                if (ratio < 0.4) {
                    //                    std::printf("Match: %d -> %d\n", firstBest.queryIdx, firstBest.trainIdx);
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

    std::unordered_map<FeatureComponent, std::unordered_set<ImageWithUniqueFeature, ImageWithUniqueFeatureHash>, FeatureComponentHash> uniqueComponents;
    for (auto const& [component, _]: featureGraph) {
        FeatureComponent const& root = findRoot(featureGraph, component);
        uniqueComponents[root].emplace(component, images[component.imageIndex].keypoints[component.featureIndex]);
    }

    std::printf("Unique components: %zu\n", uniqueComponents.size());

    for (auto const& [_, uniqueFeature]: uniqueComponents) {
        if (uniqueFeature.size() < 10) continue;

        std::printf("Images: %zu\n", uniqueFeature.size());

        for (auto& image: uniqueFeature) {
            cv::Mat out = images[image.component.imageIndex].mat.clone();
            //            cv::drawKeypoints(images[image.component.imageIndex].mat, {image.keypoint}, out);
            cv::circle(out, image.keypoint.pt, 5, {0, 0, 255}, 2);
            cv::resize(out, out, {}, 0.8, 0.8);
            cv::imshow("Keypoint", out);
            cv::waitKey();
        }
    }

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
