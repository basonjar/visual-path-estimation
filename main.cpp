#include "pch.h"
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <fstream>

#include "utils.hpp"

constexpr size_t WIDTH = 1920;
constexpr size_t HEIGHT = 1080;
// Cube
//auto K = std::make_shared<gtsam::Cal3_S2>(39.6, WIDTH, HEIGHT);
// Caterpillar
//auto K = std::make_shared<gtsam::Cal3_S2>(WIDTH * 7 / 10, WIDTH * 7 / 10, 0, WIDTH / 2, HEIGHT / 2);
// Dino Ring
auto K = std::make_shared<gtsam::Cal3_S2>(3310.4, 3325.5, 0, 316.73, 200.55);
// Temple Ring
//auto K = std::make_shared<gtsam::Cal3_S2>(1520.4, 1525.9, 0, 302.32, 246.87);
// CSGO
//auto K = std::make_shared<gtsam::Cal3_S2>(40, WIDTH, HEIGHT);

constexpr float RATIO_THRESHOLD = 0.6f;
constexpr size_t IMAGE_COUNT = 15;
const std::string DATASET = "dinoRing";

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

    auto images = imagePaths | std::views::transform([](fs::directory_entry const& imagePath) {
                      std::cout << "\tLoading " << imagePath.path() << std::endl;

                      cv::Mat mat = cv::imread(imagePath.path(), cv::IMREAD_COLOR);
                      cv::Mat descriptors;
                      std::vector<cv::KeyPoint> features;

                      cv::Ptr<cv::SIFT> descriptorExtractor = cv::SIFT::create();
                      descriptorExtractor->detectAndCompute(mat, cv::noArray(), features, descriptors);

                      return Image{mat, features, descriptors};
                  }) |
                  collect<std::vector<Image>>();
    return images;
}

int main() {
    auto imageDirectoryPath = fs::current_path() / "data" / DATASET / "images";

    // Load images and calculate features
    std::vector<Image> images = loadImages(imageDirectoryPath);

    // Create graph of connections between features across multiple images
    FeatureGraph featureGraph;

    for (size_t img1 = 0; img1 < images.size(); ++img1) {
        for (size_t img2 = img1 + 1; img2 < images.size(); ++img2) {
            using BestMatches = std::vector<cv::DMatch>;
            std::vector<BestMatches> matches;

            cv::BFMatcher matcher{cv::NORM_L2};
            matcher.knnMatch(images[img1].descriptors, images[img2].descriptors, matches, 2);

            BestMatches goodMatches;

            for (BestMatches const& bestMatch: matches) {
                assert(bestMatch.size() == 2);

                cv::DMatch const &firstBest = bestMatch[0], &secondBest = bestMatch[1];
                double ratio = firstBest.distance / secondBest.distance;

                if (ratio < RATIO_THRESHOLD) {
                    unionComponents(featureGraph, FeatureComponent(img1, firstBest.queryIdx), FeatureComponent(img2, firstBest.trainIdx));
                    goodMatches.push_back(firstBest);
                }
            }
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

    gtsam::NonlinearFactorGraph graph;

    std::unordered_set<uint64_t> imageConstraints;
    size_t smartFactorConstraintCount = 0;
    auto featureNoise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0); // sigma = pixels in u/v
    for (auto const& [featureId, imagesWithUniqueFeature]: uniqueFeatures | enumerate()) {
        auto factor = std::make_shared<SmartFactor>(featureNoise, K);
        for (auto const& image: imagesWithUniqueFeature) {
            factor->add(Point2(image.keypoint.pt.x, image.keypoint.pt.y), gtsam::Symbol('x', image.component.imageIndex));
            ++smartFactorConstraintCount;
            imageConstraints.insert(image.component.imageIndex);
        }
        graph.push_back(factor);
    }
    if (imageConstraints.size() != images.size()) throw std::runtime_error("Each image needs a constraint!");

    graph.saveGraph("graph.dot");

    pcl::visualization::PCLVisualizer viewer("3D Viewer");
    viewer.setBackgroundColor(0.16, 0.17, 0.18);
    viewer.initCameraParameters();

    gtsam::Values initial;
    std::ofstream estimates("results/" + DATASET + "/estimates.txt");

    Pose3 globalPose = Pose3::Identity();
    Pose3 newPose = Pose3::Identity();
    initial.insert(gtsam::Symbol('x', 0), globalPose);
    estimates << "1 0 0 0 1 0 0 0 1 0 0 0" << std::endl;

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
        estimates << R(0, 0) << " " << R(0, 1) <<  " " << R(0, 2) << " " << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << " " << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << " " << t(0) << " " << t(1) << " " << t(2) << std::endl;

        if (i == 1 || i == 2) {
            auto initialNoise = gtsam::noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(0.1), Vector3::Constant(0.3)).finished());// rpy (rad) then xyz (m)
            graph.addPrior(gtsam::Symbol('x', i - 1), globalPose, initialNoise);
        }
        Pose3 poseBetween{Rot3{R}, t * 0.3};

        newPose = globalPose * poseBetween;
        pcl::PointXYZ oldPoint (globalPose.translation().x(), globalPose.translation().y(), globalPose.translation().z());
        pcl::PointXYZ newPoint(newPose.translation().x(), newPose.translation().y(), newPose.translation().z());
        viewer.addLine(oldPoint, newPoint, 0.66, 0.62, 0.93, "line" + std::to_string(i));
        globalPose = newPose;

        initial.insert(gtsam::Symbol('x', i), globalPose);


        Point3 origin = globalPose* Point3{0, 0, 0};
        Point3 tip = globalPose* Point3{0, 0, 0.15};

        pcl::PointXYZ pointPcl(origin.x(), origin.y(), origin.z());
        pcl::PointXYZ pointPclTip(tip.x(), tip.y(), tip.z());

        viewer.addLine(pointPcl, pointPclTip, 0.91, 0.42, 0.53, "initial" + std::to_string(i));
        viewer.addSphere(pointPclTip, 0.006, 0.91, 0.42, 0.53, "initialTip" + std::to_string(i));
    }

    for (uint64_t i = 0; i < uniqueFeatures.size(); ++i) {
        initial.insert(gtsam::Symbol('l', i), Point3{});
    }

    gtsam::DoglegParams params;
    gtsam::Values result;
    try {
        result = gtsam::DoglegOptimizer(graph, initial, params).optimize();
    } catch (gtsam::IndeterminantLinearSystemException& e) {
        std::cout << "IndeterminantLinearSystemException: " << e.what() << std::endl;
        std::printf("Variables: %u\n", uniqueFeatures.size() + images.size());
        std::printf("SmartFactor Constraints: %zu\n", smartFactorConstraintCount);
        return 1;
    }
    result.print("result: ");

    for (uint64_t i = 0; i < uniqueFeatures.size(); ++i) {
        auto smart = std::dynamic_pointer_cast<SmartFactor>(graph[i]);
        if (!smart) continue;

        gtsam::TriangulationResult t = smart->point();
        double norm = t->x() * t->x() + t->y() * t->y() + t->z() * t->z();
        if (!t || !t.valid() || norm > 10000) continue;

        double sideLength = 0.007;
        viewer.addCube(t->x() - sideLength / 2, t->x() + sideLength / 2,
                                t->y() - sideLength / 2, t->y() + sideLength / 2,
                                t->z() - sideLength / 2, t->z() + sideLength / 2,
                                1, 1, 1, "feature" + std::to_string(i));
    }

    for(uint64_t i = 0; i < images.size() - 1; ++i) {
        auto pose1 = result.at<Pose3>(gtsam::Symbol('x', i));
        Point3 origin = pose1* Point3{0, 0, 0};
        auto pose2 = result.at<Pose3>(gtsam::Symbol('x', i + 1));
        Point3 to = pose2* Point3{0, 0, 0};

        pcl::PointXYZ pointOrigin(origin.x(), origin.y(), origin.z());
        pcl::PointXYZ pointTo(to.x(), to.y(), to.z());

        viewer.addLine(pointOrigin, pointTo, 0.56, 0.85, 0.89, "resultLine" + std::to_string(i));
    }

    std::ofstream resultsFile("results/" + DATASET + "/results.txt");

    for (uint64_t i = 0; i < images.size(); ++i) {
        auto pose = result.at<Pose3>(gtsam::Symbol('x', i));
        Matrix3 rot = pose.rotation().matrix();
        Point3 origin = pose* Point3{0, 0, 0};
        resultsFile << rot(0, 0) << " " << rot(0, 1) << " " << rot(0, 2) << " " << rot(1, 0) << " " << rot(1, 1) << " " << rot(1, 2) << " " << rot(2, 0) << " " << rot(2, 1) << " " << rot(2, 2) << " " << origin.x() << " " << origin.y() << " " << origin.z() << std::endl;
        Point3 tip = pose* Point3{0, 0, 0.15};

        pcl::PointXYZ pointPcl(origin.x(), origin.y(), origin.z());
        pcl::PointXYZ pointPclTip(tip.x(), tip.y(), tip.z());

        viewer.addLine(pointPcl, pointPclTip, 0.5, 0.83, 0.49, "result" + std::to_string(i));
        viewer.addSphere(pointPclTip, 0.006, 0.5, 0.83, 0.49, "resultTip" + std::to_string(i));
    }

    std::printf("Variables: %u\n", uniqueFeatures.size() + images.size());
    std::printf("SmartFactor Constraints: %u\n", smartFactorConstraintCount);

    std::printf("Error: %f %f\n", graph.error(result), graph.error(initial));

    viewer.spin();

    return EXIT_SUCCESS;
}
