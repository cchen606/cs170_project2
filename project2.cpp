/*
Clement Chen, cchen606@ucr.edu 862321584
cs 170 Project 2
Resources used: cplusplus.com, cppreference.com, stackoverflow.com, simplilearn.com, geeksforgeeks.org
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <numeric>

using namespace std;

double evaluation(const vector<int>& feature) {  //assuming im gonna need to pass this in when we write the actual evaluation function (for now just ignore the warning error)
    // random evaluation
    return static_cast<double>(rand() % 10000) / 100.0; // 
}


void forward_selection(int total_features) {

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
                double accuracy = evaluation(candidate_features);

                cout << "Using feature(s) { ";
                for (int f : candidate_features) cout << f << " ";
                cout << "} accuracy is " << fixed << setprecision(2) << accuracy << "%" << endl;

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
            cout << "} was best, accuracy is " << fixed << setprecision(2) << best_accuracy << "%" << endl << endl;
        }
        if(best_accuracy > overall_best_accuracy){
            overall_best_accuracy = best_accuracy;
            overall_best.push_back(current_best_feature);
        }
    }

    cout << endl << "Finished search!! The best feature subset is { ";
    for (int f : overall_best) cout << f << " ";
    cout << "}, which has an accuracy of " << fixed << setprecision(2) << overall_best_accuracy << "%" << endl;
}



void backward_elimination(int total_features) {
    vector<int> best_features(total_features);
    vector<int> overall_best;
    double overall_best_accuracy = 0.00;
    iota(best_features.begin(), best_features.end(), 1); // Initialize with all the features so we can start backwards
    double best_accuracy = evaluation(best_features);

    cout << "Using all features { ";
    for (int f : best_features) cout << f << " ";
    cout << "} accuracy is " << fixed << setprecision(2) << best_accuracy << "%" << endl;

    cout << "Beginning search. "<< endl;
    while (best_features.size() > 1) {
        vector<int> current_best_feature_set = best_features;
        double current_best_accuracy = 0.0;
        

        for (int feature : best_features) {
            vector<int> candidate_features = best_features;
            candidate_features.erase(remove(candidate_features.begin(), candidate_features.end(), feature), candidate_features.end());
            double accuracy = evaluation(candidate_features);

            cout << "Using feature(s) { ";
            for (int f : candidate_features) cout << f << " ";
            cout << "} accuracy is " << fixed << setprecision(2) << accuracy << "%" << endl;

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
            cout << "} was best, accuracy is " << fixed << setprecision(2) << best_accuracy << "%" << endl;
        }
        if (best_accuracy > overall_best_accuracy) {
                overall_best_accuracy = best_accuracy;
                overall_best = best_features;
        }
    }

    cout << "Finished search!! The best feature subset is { ";
    for (int f : overall_best) cout << f << " ";
    cout << "}, which has an accuracy of " << fixed << setprecision(2) << overall_best_accuracy << "%" << endl;
}



int main(){
    int total_features;
    int choice;
    cout << "Welcome to Clement Chen's Feature Selection Algorithm." << endl;
    cout << "Please enter total number of features: ";
    cin >> total_features;


        cout << "Type the number of the algorithm you want to run. " << endl << endl;
        cout << "1. Forward Selection" << endl << "2. Backward Elimination" << endl << "3. Special (implement ltr)" << endl;
        
        cin >> choice;

        if (choice == 1) {
            forward_selection(total_features);
        } else if (choice == 2) {
            backward_elimination(total_features);
        } else {
            cout << "Invalid choice!" << endl;
        }

    return 0;
}