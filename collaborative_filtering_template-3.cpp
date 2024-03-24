#include <iostream>
#include <fstream>
#include <istream>
#include <random>
#include <set>
#include <map>
#include <vector>
#include <sstream>
//#include <cmath>

using namespace std;

bool toss_coin(double p) {
    // Return true with probability p, false with probability 1-p
    static std::default_random_engine engine(std::random_device{}());
    std::bernoulli_distribution distribution(p);
    return distribution(engine);
}

double generate_uniform_random_number() {
    // Static to maintain state across function calls and only initialized once
    static std::default_random_engine engine(std::random_device{}());
    std::uniform_real_distribution<double> distribution(0.0, 1);

    // Generate and return the random number
    return distribution(engine);
}

double dot_product(std::vector<double> &v1, std::vector<double> &v2) {
    double result = 0;
    for (int i = 0; i < v1.size(); i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

int main() {

    std::ifstream file("datasets/movie-ratings-small.csv");
    std::string line;
    std::map<std::pair<int, int>, double> ratings;
    std::map<std::pair<int, int>, double> test_set;
    std::map<int, std::set<int>> users_movies; 
    std::map<int, std::set<int>> movies_users;
    std::set<int> users;
    std::set<int> movies;
    
    int K = 15; // number of latent dimensions
    int m = 2000; // upper bound for number of users
    int n = 2000; // upper bound number of movies
    
    double test_set_size = 0.1; // percentage of the data will be used for testing
    double lambda = 1e-3; // regularization parameter
    double eta = 1e-4; // learning rate
    double decay = 0.9; // decay rate
    int n_iterations = 35; // number of iterations for the gradient descent

    if (file.is_open()) {
        std::getline(file, line); // skip the first line

        while (std::getline(file, line)) {

            std::istringstream iss(line);
            std::string token;
            // read user, movie, and rating
            std::getline(iss, token, ',');
            int user = std::stol(token);
            std::getline(iss, token, ',');
            int movie = std::stol(token);
            std::getline(iss, token, ',');
            double rating = std::stod(token);

            if (toss_coin(1 - test_set_size)) {
                // if the coin toss is true, add the rating to the training set
                ratings[std::make_pair(user, movie)] = rating;
                users_movies[user].insert(movie); // add song to user's list of songs
                movies_users[movie].insert(user); // add user to song's list of users
            } else {
                // if the coin toss is false, add the rating to the test set
                test_set[std::make_pair(user, movie)] = rating;
            }
             
            // keep track of users and movies that have been added
            // the Ids might be larger than the number of users and movies
            users.insert(user); 
            movies.insert(movie);
        }

        file.close();
    } else {
        std::cout << "Unable to open file" << std::endl;
    }

    std::cout << "Finish Reading File" << std::endl;

    // initialize U and V for the collaborative filtering
    std::vector<std::vector<double>> U(m, std::vector<double>(K, 0));
    std::vector<std::vector<double>> V(n, std::vector<double>(K, 0));

    // initialize U and V with random values
    for (int i : users) {
        for (int k = 0; k < K; k++) {
            U[i][k] = generate_uniform_random_number();
        }
    }

    for (int j : movies) {
        for (int k = 0; k < K; k++) {
            V[j][k] = generate_uniform_random_number();
        }
    }

    for (int t = 0; t < n_iterations; t++) {
        eta = eta * decay; // decay the learning rate over time

        // implement gradient descent here:
        // you may want to use for (int i : users) and for (int j : movies) 
        // to iterate over all users and movies instead of for (int i = 0; i < m; i++) and for (int j = 0; j < n; j++)
        // to avoid iterating over users and movies that are not in the training set

        cout << "Finished iteration " << t << endl;
    }

    std::cout << "Finish Gradient Descent" << std::endl;
    
    // calculate the mean absolute error
    double mae = 0;
    double mae_random = 0; // mean absolute error if we were to guess 3 for every rating

    for (auto &x : test_set) {
        int i = x.first.first;
        int j = x.first.second;
        double r = x.second;
        double prediction = dot_product(U[i], V[j]);
        if (prediction > 5) {
            prediction = 5;
        } else if (prediction < 1) {
            prediction = 1;
        }
        mae += abs(dot_product(U[i], V[j]) - r);
        mae_random += abs(3 - r);
    }

    mae = static_cast<double>(mae / test_set.size());
    mae_random = static_cast<double>(mae_random / test_set.size());
    std::cout << "Mean Absolute Error: " << mae << std::endl;
    std::cout << "Mean Absolute Error Random Guess: " << mae_random << std::endl;

    return 0;
}

