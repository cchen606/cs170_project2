/*
Clement Chen, cchen606@ucr.edu 862321584
cs 170 Project 2
Resources used: cplusplus.com, cppreference.com, stackoverflow.com, geeksforgeeks.org
*/


#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <chrono>

using namespace std;

double calcDistance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
}

void normalize(vector<vector<double>>& dataset) {
    unsigned int numFeatures = dataset[0].size() - 1; // Exclude the class label
    for (unsigned int j = 1; j <= numFeatures; ++j) { // Iterate through each feature column
        double minVal = dataset[0][j];
        double maxVal = dataset[0][j];
        // Find min and max for the current feature
        for (unsigned int i = 0; i < dataset.size(); ++i) {
            minVal = min(minVal, dataset[i][j]);
            maxVal = max(maxVal, dataset[i][j]);
        }
        // Normalize the feature values to range [0, 1]
        for (unsigned int i = 0; i < dataset.size(); ++i) {
            dataset[i][j] = (dataset[i][j] - minVal) / (maxVal - minVal);
        }
    }
}

// Classifier Class
class NNClassifier {
public:
    vector<vector<double>> trainingData;
    // train
    void train(const vector<vector<double>>& data) {
        trainingData = data;
    }

    // test
    int test(const vector<double>& instance, const vector<int>& featureSubset) {
        double minDistance = numeric_limits<double>::max();
        int predictedClass = -1;

        for (const auto& trainInstance : trainingData) {
            vector<double> trainFeatures;
            vector<double> testFeatures;

            for (int feature : featureSubset) {
                trainFeatures.push_back(trainInstance[feature]);
                testFeatures.push_back(instance[feature]);
            }

            double distance = calcDistance(trainFeatures, testFeatures);
            if (distance < minDistance) {
                minDistance = distance;
                predictedClass = static_cast<int>(trainInstance[0]);
            }
        }

        return predictedClass;
    }
};

// Validator Class
class Validator {
public:
    double validate(NNClassifier& classifier, const vector<vector<double>>& dataset, const vector<int>& featureSubset) {
        int correct = 0;

        for (size_t i = 0; i < dataset.size(); ++i) {
            vector<vector<double>> trainingData = dataset;
            vector<double> testInstance = trainingData[i];
            trainingData.erase(trainingData.begin() + i);

            classifier.train(trainingData);
            int predictedClass = classifier.test(testInstance, featureSubset);

            if (i % 10 == 0) {
                cout << "Instance " << i + 1 << ": Predicted Class = " << predictedClass 
                     << ", Actual Class = " << static_cast<int>(testInstance[0]) 
                     << " => " << (predictedClass == static_cast<int>(testInstance[0]) ? "Correct" : "Incorrect") 
                     << endl;
            }


            if (predictedClass == static_cast<int>(testInstance[0])) {
                ++correct;
            }
        }

        return static_cast<double>(correct) / dataset.size();
    }
};

vector<vector<double>> loadDataset(const string& filename) {
    vector<vector<double>> dataset;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> instance;
        double value;
        while (ss >> value) {
            instance.push_back(value);
        }
        dataset.push_back(instance);
    }

    return dataset;
}

int main() {
    string smallDatasetFile = "small-test-dataset.txt";
    string largeDatasetFile = "large-test-dataset.txt";

    vector<vector<double>> smallDataset = loadDataset(smallDatasetFile);
    vector<vector<double>> largeDataset = loadDataset(largeDatasetFile);

    normalize(smallDataset);
    normalize(largeDataset);

    // test subsets accuaracy should be about 0.89 and 0.949 respectively
    vector<int> smallFeatureSubset = {3, 5, 7};
    vector<int> largeFeatureSubset = {1, 15, 27};

    NNClassifier classifier;
    Validator validator;

    // test small dataset.txt
    auto startSmall = chrono::high_resolution_clock::now();
    double smallAccuracy = validator.validate(classifier, smallDataset, smallFeatureSubset);
    auto endSmall = chrono::high_resolution_clock::now();

    cout << "Small Dataset Accuracy: " << fixed << setprecision(2) << smallAccuracy * 100 << "%" << endl;
    cout << "Time Taken: " << chrono::duration_cast<chrono::milliseconds>(endSmall - startSmall).count() << "ms\n";

    // test largedataset.txt
    auto startLarge = chrono::high_resolution_clock::now();
    double largeAccuracy = validator.validate(classifier, largeDataset, largeFeatureSubset);
    auto endLarge = chrono::high_resolution_clock::now();

    cout << "Large Dataset Accuracy: " << fixed << setprecision(2) << largeAccuracy * 100 << "%" << endl;
    cout << "Time Taken: " << chrono::duration_cast<chrono::milliseconds>(endLarge - startLarge).count() << "ms\n";

    return 0;
}
