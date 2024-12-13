/*
Clement Chen, cchen606@ucr.edu 862321584
cs 170 Project 2 Final submission (part 3)
Resources used: cplusplus.com, cppreference.com, stackoverflow.com, geeksforgeeks.org
*/
//Results at the bottom

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>

using namespace std;

// Calculate Euclidean distance
double calcDistance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
}

// Normalize dataset
void normalize(vector<vector<double>>& dataset) {
    unsigned int numFeatures = dataset[0].size() - 1; // Exclude the class label
    for (unsigned int j = 1; j <= numFeatures; ++j) { 
        double minVal = dataset[0][j];
        double maxVal = dataset[0][j];
        // Find min and max
        for (unsigned int i = 0; i < dataset.size(); ++i) {
            minVal = min(minVal, dataset[i][j]);
            maxVal = max(maxVal, dataset[i][j]);
        }
        // Normalize  [0, 1]
        for (unsigned int i = 0; i < dataset.size(); ++i) {
            dataset[i][j] = (dataset[i][j] - minVal) / (maxVal - minVal);
        }
    }
}

// Classifier Class
class NNClassifier {
public:
    vector<vector<double>> trainingData;
    // Train
    void train(const vector<vector<double>>& data) {
        trainingData = data;
    }

    // Test
    int test(const vector<double>& instance, const vector<int>& featureSubset, size_t skipIndex = -1) {
    double minDistance = numeric_limits<double>::max();
    int predictedClass = -1;

    for (size_t i = 0; i < trainingData.size(); ++i) {
        if (i == skipIndex) continue; // Skip test instance

        vector<double> trainFeatures, testFeatures;
        for (int feature : featureSubset) {
            trainFeatures.push_back(trainingData[i][feature]);
            testFeatures.push_back(instance[feature]);
        }

        double distance = calcDistance(trainFeatures, testFeatures);
        if (distance < minDistance) {
            minDistance = distance;
            predictedClass = static_cast<int>(trainingData[i][0]);
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
    classifier.train(dataset); 

    for (size_t i = 0; i < dataset.size(); ++i) {
        int predictedClass = classifier.test(dataset[i], featureSubset, i);
        if (predictedClass == static_cast<int>(dataset[i][0])) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / dataset.size();
}
};

// Load dataset
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

// Feature Selection Functions
void forward_selection(int total_features, vector<vector<double>>& dataset, NNClassifier& classifier, Validator& validator) {
    vector<int> best_features;
    double best_accuracy = 0.00;
    vector<int> overall_best;
    double overall_best_accuracy = 0.00;

    cout << "Beginning search..." << endl;
    for (int i = 1; i <= total_features; ++i) {
        int current_best_feature = -1;
        double current_best_accuracy = 0.00;

        for (int feature = 1; feature <= total_features; ++feature) {
            if (find(best_features.begin(), best_features.end(), feature) == best_features.end()) {
                vector<int> candidate_features = best_features;
                candidate_features.push_back(feature);
                double accuracy = validator.validate(classifier, dataset, candidate_features);

                cout << "Using feature(s) { ";
                for (int f : candidate_features) cout << f << " ";
                cout << "} accuracy is " << fixed << setprecision(2) << accuracy * 100.0 << "%" << endl;

                if (accuracy > current_best_accuracy) {
                    current_best_accuracy = accuracy;
                    current_best_feature = feature;
                }
            }
        }

        if (current_best_feature != -1) {
            best_features.push_back(current_best_feature);
            best_accuracy = current_best_accuracy;

            cout << "Feature set { ";
            for (int f : best_features) cout << f << " ";
            cout << "} was best, accuracy is " << fixed << setprecision(2) << best_accuracy * 100.0 << "%" << endl;
        }
        if(best_accuracy > overall_best_accuracy){
            overall_best_accuracy = best_accuracy;
            overall_best.push_back(current_best_feature);
        }
    }

    cout << endl << "Finished search!! The best feature subset is { ";
    for (int f : overall_best) cout << f << " ";
    cout << "}, which has an accuracy of " << fixed << setprecision(2) << overall_best_accuracy * 100.0 << "%" << endl;
}




void backward_elimination(int total_features, vector<vector<double>>& dataset, NNClassifier& classifier, Validator& validator) {
    vector<int> best_features(total_features);
    vector<int> overall_best;
    double overall_best_accuracy = 0.00;
    iota(best_features.begin(), best_features.end(), 1); // Initialize with all the features so we can start backwards
    double best_accuracy = validator.validate(classifier, dataset, best_features);

    cout << "Using all features { ";
    for (int f : best_features) cout << f << " ";
    cout << "} accuracy is " << fixed << setprecision(2) << best_accuracy * 100.0 << "%" << endl;

    cout << "Beginning search. "<< endl;
    while (best_features.size() > 1) {
        vector<int> current_best_feature_set = best_features;
        double current_best_accuracy = 0.0;
        

        for (int feature : best_features) {
            vector<int> candidate_features = best_features;
            candidate_features.erase(remove(candidate_features.begin(), candidate_features.end(), feature), candidate_features.end());
            double accuracy = validator.validate(classifier, dataset, candidate_features);

            cout << "Using feature(s) { ";
            for (int f : candidate_features) cout << f << " ";
            cout << "} accuracy is " << fixed << setprecision(2) << accuracy * 100.0 << "%" << endl;

            if (accuracy > current_best_accuracy) {
                current_best_accuracy = accuracy;
                current_best_feature_set = candidate_features;
            }
        }

        if (!current_best_feature_set.empty()) {
            best_features = current_best_feature_set;
            best_accuracy = current_best_accuracy;

            cout << "Feature set { ";
            for (int f : best_features) cout << f << " ";
            cout << "} was best, accuracy is " << fixed << setprecision(2) << best_accuracy * 100.0 << "%" << endl;
        }
        if (best_accuracy > overall_best_accuracy) {
                overall_best_accuracy = best_accuracy;
                overall_best = best_features;
        }
    }

    cout << "Finished search!! The best feature subset is { ";
    for (int f : overall_best) cout << f << " ";
    cout << "}, which has an accuracy of " << fixed << setprecision(2) << overall_best_accuracy * 100.0 << "%" << endl;
}




int main() {
    int total_features;
    string dataset_filename;

    cout << "Welcome to Clement Chen's Feature Selection Algorithm." << endl;
    cout << "Type in the name of the file to test: ";
    cin >> dataset_filename;

    vector<vector<double>> dataset = loadDataset(dataset_filename);
    normalize(dataset);

    total_features = dataset[0].size() - 1; // Exclude the class label

    cout << "Type the number of the algorithm you want to run: " << endl;
    cout << "1. Forward Selection" << endl << "2. Backward Elimination" << endl;

    int choice;
    cin >> choice;

    NNClassifier classifier;
    Validator validator;

    if (choice == 1) {
        forward_selection(total_features, dataset, classifier, validator);
    }
    else if (choice == 2) {
        backward_elimination(total_features, dataset, classifier, validator);
    } 
    else {
        cout << "Invalid choice!" << endl;
    }

    return 0;
}


// Results
// - Group: Clement Chen – cchen606
// - Small Dataset Results:
// - Forward: Feature Subset: {5, 3}, Acc: 92.0%
// - Backward: Feature Subset: {3, 5} Acc: 92.0%
// - Large Dataset Results:
// - Forward: Feature Subset: {27, 1}, Acc: 95.5%
// - Backward: Feature Subset: {27, 1}, Acc: 95.5%
// - Titanic Dataset Results:
// - Forward: Feature Subset: {2}, Acc: 78.01%
// - Backward: Feature Subset: {2}, Acc: 78.01%
