#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/ProjectionFactor.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/SVD>

auto K = std::make_shared<gtsam::Cal3_S2>(960, 540, 0, 1344, 1344);

namespace fs = std::filesystem;

using gtsam::Matrix3;
using gtsam::Matrix4;
using gtsam::Point2;
using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::Vector;
using gtsam::Vector3;

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
    std::ranges::copy(fs::directory_iterator(imageDirectoryPath), std::back_inserter(imagePaths));
    //    for (auto const& entry: fs::directory_iterator(imageDirectoryPath)) {
    //        imagePaths.push_back(entry);
    //    }
    std::ranges::sort(imagePaths, [](fs::directory_entry const& a, fs::directory_entry const& b) {
        return a.path().stem().string() < b.path().stem().string();
    });
    imagePaths.resize(30);

    for (fs::directory_entry const& imagePath: imagePaths) {
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

    using ImagesWithUniqueFeature = std::unordered_set<ImageWithUniqueFeature, ImageWithUniqueFeatureHash>;
    std::unordered_map<FeatureComponent, ImagesWithUniqueFeature, FeatureComponentHash> uniqueComponents;
    //    std::unordered_set<FeatureComponent, FeatureComponentHash> seenComponents;
    std::vector<ImagesWithUniqueFeature> uniqueFeatures;
    for (auto const& [component, _]: featureGraph) {
        FeatureComponent const& root = findRoot(featureGraph, component);
        uniqueComponents[root].emplace(component, images[component.imageIndex].keypoints[component.featureIndex]);
    }
    std::ranges::copy(uniqueComponents | std::views::values, std::back_inserter(uniqueFeatures));

    std::printf("Unique features: %zu\n", uniqueFeatures.size());

    //    for (auto const& [_, uniqueFeature]: uniqueComponents) {
    //        if (uniqueFeature.size() != 2) continue;
    //
    //        std::printf("Images: %zu\n", uniqueFeature.size());
    //
    //        for (auto& image: uniqueFeature) {
    //            cv::Mat out = images[image.component.imageIndex].mat.clone();
    //            //            cv::drawKeypoints(images[image.component.imageIndex].mat, {image.keypoint}, out);
    //            cv::circle(out, image.keypoint.pt, 5, {0, 0, 255}, 2);
    //            cv::resize(out, out, {}, 0.8, 0.8);
    //            cv::imshow("Keypoint", out);
    //            cv::waitKey();
    //        }
    //    }

    gtsam::NonlinearFactorGraph graph;

    auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas(
            (Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.3))
                    .finished());// 30cm std on x,y,z 0.1 rad on roll,pitch,yaw
    graph.addPrior(gtsam::Symbol('x', 0), Pose3{}, poseNoise);

    for (auto const& [featureId, uniqueFeature]: std::views::zip(std::views::iota(0uz), uniqueFeatures)) {
        for (auto const& image: uniqueFeature) {
            //            std::printf("Feature %zu in image %zu\n", featureId, image.component.imageIndex);
            graph.emplace_shared<gtsam::GenericProjectionFactor<gtsam::Pose3, gtsam::Point3, gtsam::Cal3_S2>>(
                    Point2(image.keypoint.pt.x, image.keypoint.pt.y),
                    gtsam::noiseModel::Isotropic::Sigma(2, 1.0),
                    gtsam::Symbol('x', image.component.imageIndex),
                    gtsam::Symbol('l', featureId),
                    K);
        }
    }

    auto pointNoise = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
    graph.addPrior(gtsam::Symbol('l', 0), Point3{}, pointNoise);

    gtsam::Values initial;

    Pose3 initialPose{};
    initial.insert(gtsam::Symbol('x', 0), initialPose);
    for (size_t i = 1; i < images.size(); ++i) {
        Image const& image1 = images[i - 1];
        Image const& image2 = images[i];

        std::vector<cv::Point2f> points1;
        std::ranges::copy(image1.keypoints | std::views::transform([](cv::KeyPoint const& keypoint) { return keypoint.pt; }), std::back_inserter(points1));
        std::vector<cv::Point2f> points2;
        std::ranges::copy(image2.keypoints | std::views::transform([](cv::KeyPoint const& keypoint) { return keypoint.pt; }), std::back_inserter(points2));

        cv::Mat Kcv;
        cv::eigen2cv(K->K(), Kcv);
        cv::Mat Ecv = cv::findEssentialMat(points1, points2, Kcv, cv::RANSAC, 0.999, 1.0);

        Matrix3 E;
        cv::cv2eigen(Ecv, E);

        Eigen::JacobiSVD<Matrix3> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Matrix3 U = svd.matrixU();
        Matrix3 V = svd.matrixV();

        Vector3 t = U.col(2);
        Rot3 R{U * V.transpose()};
        Pose3 pose{R, t};

        std::cout << "Pose: " << pose << std::endl;

        initialPose = initialPose.compose(pose);
        initial.insert(gtsam::Symbol('x', i), initialPose);
    }

    for (size_t i = 0; i < uniqueFeatures.size(); ++i) {
        initial.insert(gtsam::Symbol('l', i), Point3{});
    }

    gtsam::Values result = gtsam::DoglegOptimizer(graph, initial).optimize();
    result.print("result: ");

    return EXIT_SUCCESS;
}
