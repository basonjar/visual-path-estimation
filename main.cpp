#include "pch.h"
#include <gtsam/slam/SmartProjectionPoseFactor.h>

#include "utils.hpp"

constexpr size_t WIDTH = 1920;
constexpr size_t HEIGHT = 1080;
auto K = std::make_shared<gtsam::Cal3_S2>(WIDTH * 7 / 10, WIDTH * 7 / 10, 0, WIDTH / 2, HEIGHT / 2);
constexpr float RATIO_THRESHOLD = 0.55f;
constexpr size_t IMAGE_COUNT = 5;
constexpr size_t IMAGE_STEP = 1;

namespace fs = std::filesystem;

using gtsam::Matrix3;
using gtsam::Matrix4;
using gtsam::Point2;
using gtsam::Point3;
using gtsam::Pose3;
using gtsam::Rot3;
using gtsam::Vector;
using gtsam::Vector3;
using SmartFactor = gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2>;

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

[[nodiscard]] FeatureComponent& findRoot(FeatureGraph& graph, FeatureComponent const& component) {// NOLINT(misc-no-recursion)
    auto it = graph.find(component);
    if (it == graph.end()) return graph[component] = component;

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

    auto imagePaths = fs::directory_iterator(imageDirectoryPath) | collect<std::vector<fs::directory_entry>>();
    std::ranges::sort(imagePaths, [](fs::directory_entry const& a, fs::directory_entry const& b) {
        return a.path().stem().string() < b.path().stem().string();
    });
    imagePaths.resize(IMAGE_COUNT);

    std::vector<fs::directory_entry> newImagePaths;
    for (int i = 0; i < imagePaths.size(); i += IMAGE_STEP) {
        newImagePaths.push_back(imagePaths[i]);
    }

    auto images = newImagePaths | std::views::transform([](fs::directory_entry const& imagePath) {
                      std::cout << "\tLoading " << imagePath.path() << std::endl;

                      cv::Mat mat = cv::imread(imagePath.path(), cv::IMREAD_COLOR);
                      cv::Mat descriptors;
                      std::vector<cv::KeyPoint> features;

                      cv::Ptr<cv::ORB> descriptorExtractor = cv::ORB::create();
                      descriptorExtractor->detectAndCompute(mat, cv::noArray(), features, descriptors);

                      return Image{mat, features, descriptors};
                  }) |
                  collect<std::vector<Image>>();
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

            using BestMatches = std::vector<cv::DMatch>;
            std::vector<BestMatches> matches;

            cv::BFMatcher matcher{cv::NORM_HAMMING};
            matcher.knnMatch(images[img1].descriptors, images[img2].descriptors, matches, 2);

            BestMatches goodMatches;

            for (BestMatches const& bestMatch: matches) {
                assert(bestMatch.size() == 2);

                cv::DMatch const &firstBest = bestMatch[0], &secondBest = bestMatch[1];
                double ratio = firstBest.distance / secondBest.distance;

                if (ratio < RATIO_THRESHOLD) {
                    //                    std::printf("Match: %d -> %d\n", firstBest.queryIdx, firstBest.trainIdx);
                    unionComponents(featureGraph, FeatureComponent(img1, firstBest.queryIdx), FeatureComponent(img2, firstBest.trainIdx));
                    goodMatches.push_back(firstBest);
                }
            }

//                        cv::Mat m;
//                        cv::drawMatches(images[img1].mat, images[img1].keypoints,
//                                        images[img2].mat, images[img2].keypoints,
//                                        goodMatches, m);
//                        cv::resize(m, m, {}, 0.75, 0.75);
//                        cv::imshow("Matches", m);
//                        cv::waitKey(0);
        }
    }

    using ImagesWithUniqueFeature = std::unordered_set<ImageWithUniqueFeature, ImageWithUniqueFeatureHash>;
    std::unordered_map<FeatureComponent, ImagesWithUniqueFeature, FeatureComponentHash> uniqueComponents;
    for (auto const& [component, _]: featureGraph) {
        FeatureComponent const& root = findRoot(featureGraph, component);
        uniqueComponents[root].emplace(component, images[component.imageIndex].keypoints[component.featureIndex]);
    }
    auto uniqueFeatures = uniqueComponents | std::views::values | collect<std::vector<ImagesWithUniqueFeature>>();

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

    // Add a prior on pose x1. This indirectly specifies where the origin is.
    graph.addPrior(gtsam::Symbol('x', 0), Pose3::Identity(), gtsam::noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.3)).finished()));

    auto poseNoise = gtsam::noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(M_PI_4), Vector3::Constant(10)).finished());// rpy (rad) then xyz (m)
    for (uint64_t i = 1; i < images.size(); ++i) {
        graph.emplace_shared<gtsam::BetweenFactor<Pose3>>(
                gtsam::Symbol('x', i - 1), gtsam::Symbol('x', i),
                Pose3::Identity(),
                poseNoise);
    }

    std::unordered_set<uint64_t> imageConstraints;

    auto featureNoise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0); // 2 pixels in u/v
    for (auto const& [featureId, imagesWithUniqueFeature]: uniqueFeatures | enumerate()) {
        printf("This many features: %zu\n", imagesWithUniqueFeature.size());
//        if (imagesWithUniqueFeature.size() < 3) continue;
        SmartFactor::shared_ptr factor(new SmartFactor(featureNoise, K));
        for (auto const& image: imagesWithUniqueFeature) {
            factor->add(Point2(image.keypoint.pt.x, image.keypoint.pt.y), image.component.imageIndex);

//                        cv::Mat m;
//                        cv::drawKeypoints(images[image.component.imageIndex].mat, {image.keypoint}, m);
//                        cv::resize(m, m, {}, 0.8, 0.8);
//                        cv::imshow("Keypoint", m);
//                        cv::waitKey();

            //            std::printf("Feature %zu in image %zu\n", featureId, image.component.imageIndex);


            imageConstraints.insert(image.component.imageIndex);
        }
    }
    if (imageConstraints.size() != images.size()) throw std::runtime_error("Each image needs a constraint!");

    graph.saveGraph("graph.dot");

    pcl::visualization::PCLVisualizer viewer("3D Viewer");
    viewer.setBackgroundColor(0, 0, 0);
    viewer.addCoordinateSystem(1.0);
    viewer.initCameraParameters();

    gtsam::Values initial;

    Pose3 globalPose = Pose3::Identity();
    initial.insert(gtsam::Symbol('x', 0), globalPose);

    for (uint64_t i = 1; i < images.size(); ++i) {
        Image const& image1 = images[i - 1];
        Image const& image2 = images[i];

        cv::BFMatcher matcher{cv::NORM_L2, true};
        std::vector<cv::DMatch> matches;
        matcher.match(image1.descriptors, image2.descriptors, matches);

        auto points1 = matches | std::views::transform([&](auto const& match) { return image1.keypoints[match.queryIdx].pt; }) | collect<std::vector<cv::Point2f>>();
        auto points2 = matches | std::views::transform([&](auto const& match) { return image2.keypoints[match.trainIdx].pt; }) | collect<std::vector<cv::Point2f>>();

        cv::Mat Kcv;
        cv::eigen2cv(K->K(), Kcv);

        cv::Mat Ecv = cv::findEssentialMat(points1, points2, Kcv, cv::RANSAC, 0.999, 1.0);
        cv::Mat Rcv, tcv;
        cv::recoverPose(Ecv, points1, points2, Kcv, Rcv, tcv);

        Matrix3 R;
        cv::cv2eigen(Rcv, R);
        Point3 t;
        cv::cv2eigen(tcv, t);

        //        std::cout << "IMAGE DELTA:" << std::endl;
        //        std::cout << "Translation: " << t << std::endl;
        //        std::cout << "Rotation: " << R.eulerAngles(0, 1, 2) << std::endl;

        Pose3 poseBetween{Rot3{R}, t * 0.3};

        globalPose = globalPose * poseBetween;

        initial.insert(gtsam::Symbol('x', i), globalPose);

        Point3 origin = globalPose* Point3{0, 0, 0};
        Point3 tip = globalPose* Point3{0, 0, 1};

        pcl::PointXYZ pointPcl(origin.x(), origin.y(), origin.z());
        pcl::PointXYZ pointPclTip(tip.x(), tip.y(), tip.z());

        //        viewer.addSphere(pointPcl, 0.1, 1.0, 0, 0, name);
        viewer.addLine(pointPcl, pointPclTip, 1, 0, 0, "arrow" + std::to_string(i));
        viewer.addText3D(std::to_string(i), pointPcl, 0.15, 0, 0, 1, std::to_string(i));
    }

    for (uint64_t i = 0; i < uniqueFeatures.size(); ++i) {
        initial.insert(gtsam::Symbol('l', i), Point3{});
    }

    //    viewer.spin();

    gtsam::DoglegParams params;
    gtsam::Values result = gtsam::DoglegOptimizer(graph, initial, params).optimize();
    result.print("result: ");

    return EXIT_SUCCESS;
}
