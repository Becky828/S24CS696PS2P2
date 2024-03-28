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

double dot_product(std::vector<double>& v1, std::vector<double>& v2) {
	double result = 0;
	for (int i = 0; i < v1.size(); i++) {
		result += v1[i] * v2[i];
	}
	return result;
}


//Put custom functions here
//int derived_u_getter(std::vector<std::vector<double>> u) {
std::vector<std::vector<double>> derived_u_getter(int m, int K, double lambda, std::vector<std::vector<double>> U, std::set<int> users) {
	std::vector<std::vector<double>> regularized_U(m, std::vector<double>(K, 0));
	for (int i : users) {
		for (int k = 0; k < K; k++) {
			regularized_U[i][k] = 2 * lambda * U[i][k];
		}
	}
	return regularized_U;
	//return current_derived_u_regularized_norm + current_u_regularization;
}

//int derived_v_getter(std::vector<std::vector<double>> v) {
std::vector<std::vector<double>> derived_v_getter(int n, int K, double lambda, std::vector<std::vector<double>> V, std::set<int> movies) {
	std::vector<std::vector<double>> regularized_V(n, std::vector<double>(K, 0));
	for (int j : movies) {
		for (int k = 0; k < K; k++) {
			regularized_V[j][k] = 2 * lambda * V[j][k];
		}
	}
	return regularized_V;
	//double current_v_regularization = 2 * lambda * current_v;
	//return current_derived_v_regularized_norm + current_v_regularization;
}

//Put V transposer here
std::vector<std::vector<double>>  v_transposer(int K, std::vector<std::vector<double>> V, std::set<int> movies) {
	int n = V.size();
	std::vector<std::vector<double>> V_transposed(n, std::vector<double>(movies.size() + 1, 0));
	//std::vector<std::vector<double>> V_transposed(std::vector<double>(K, 0),m);

	/*for (int j = 0; j < m_size; j++) {
		for (int k = 0; k < K; k++) {
			V_transposed[k][j] = V[j][k];
		}
	}*/


	for (int j : movies) {
		for (int k = 0; k < K; k++) {
			V_transposed[k][j] = V[j][k];
		}
	}

	return V_transposed;
}
//Put u dot transposed v here

//Put ratings difference here

//Put derived u product here

//Put derived v product here

//Put gradient u here

//Put gradient v here

//Put stochastic u gradient here

//Put stochastic v gradient here

//Put u gradient descent here

//Put v gradient descent here

//Put stochastic u gradient descent here

//Put stochastic v gradient descent here


int main() {

	//std::ifstream file("datasets/movie-ratings-small.csv");
	std::ifstream file("Dataset.csv");
	// std::ifstream file("Movie_Id_Titles.csv");    
	 //std::ifstream file("ratings.csv");
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
				double current_rating_a = ratings[std::make_pair(user, movie)];
				users_movies[user].insert(movie); // add movie to user's list of movies
				auto user_movie = users_movies[user];
				//std::map<int, std::set<int>>::iterator it = users_movies.find(user);

				//auto current_movie = it->second.begin();
				//int current_movie = users_movies.find(user);
				//int current_movie = users_movies.at(user).;
				movies_users[movie].insert(user); // add user to movie's list of users
			}
			else {
				// if the coin toss is false, add the rating to the test set
				test_set[std::make_pair(user, movie)] = rating;
			}

			// keep track of users and movies that have been added
			// the Ids might be larger than the number of users and movies
			users.insert(user);
			movies.insert(movie);
		}

		file.close();
	}
	else {
		std::cout << "Unable to open file" << std::endl;
	}

	std::cout << "Finish Reading File" << std::endl;

	// initialize U and V for the collaborative filtering
	std::vector<std::vector<double>> U(m, std::vector<double>(K, 0));
	std::vector<std::vector<double>> V(n, std::vector<double>(K, 0));


	std::vector<std::vector<double>> derived_norm_U(m, std::vector<double>(K, 0));
	std::vector<std::vector<double>> derived_norm_V(n, std::vector<double>(K, 0));


	std::vector<std::vector<double>> ratings_difference_U_product(m, std::vector<double>(K, 0));
	std::vector<std::vector<double>> ratings_difference_V_product(n, std::vector<double>(K, 0));

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

	int base_gradient_U;
	int base_gradient_V;
	//std::vector<std::vector<double>> V(n, std::vector<double>(K, 0));
	int m_size = movies.size();


	//initialize shared derived variables

	//U
	//double derived_norm_u = 0;
	double derived_regularized_loss_u = 0;

	//V
	//double derived_norm_v = 0;
	double derived_regularized_loss_v = 0;

	for (int t = 0; t < n_iterations; t++) {
		eta = eta * decay; // decay the learning rate over time

		// implement gradient descent here:
		// you may want to use for (int i : users) and for (int j : movies) 
		// to iterate over all users and movies instead of for (int i = 0; i < m; i++) and for (int j = 0; j < n; j++)
		// to avoid iterating over users and movies that are not in the training set

		// you may also want to use the dot_product function to calculate the dot product of U[i] and V[j]
		// and the derived_u_getter and derived_v_getter functions to calculate the sum of the derived U and V values
		// you can also use the lambda, eta, and decay variables
		//derived_norm_u = 0;
		//derived_regularized_loss_u = 0;

		std::vector<std::vector<double>> V_transposed = v_transposer(K, V, movies);

		//Initialize U variables for gradient descent

		derived_norm_U = derived_u_getter(m, K, lambda, U, users);

		for (int i : users) {
			base_gradient_U = 0;
			int current_user = i;
			std::set<int> current_user_movie_set = users_movies[i];
		
	for (int j : current_user_movie_set) {
				int current_movie = j;
				double current_rating = ratings[std::make_pair(i, j)];
				auto current_U = U[i];
				auto current_V_transposed = V_transposed[j];
				double U_dot_V_transposed = dot_product(current_U, current_V_transposed);
				double ratings_difference = U_dot_V_transposed - current_rating;
				
				for (int j : movies) {
						for (int k = 0; k < K; k++) {
							auto current_V = V[j][k];
							ratings_difference_V_product[j][k] = ratings_difference * current_V;
						}
				}
				//std::vector<std::vector<double>>ratings_difference_V_product = ratings_difference * current_V;
				// 
				//base_gradient_U = base_gradient_U + ratings_difference_V_product;
			
	}
			
			//int current_movie = current_user_movie;
			//for (int k = 0; k < K; k++) {
				//derived_norm_u = derived_u_getter(m, K, lambda, U, users);
			//}
			//derived_regularized_gradient_descent_u = gradient_descent_u_utility(eta, ratings, derived_norm_u, V_transposed);
			//double U_dot_V_transposed = dot_product(U[i], V_transposed[i]);


			//int current_movie = users_movies[i];
			//double current_rating = ratings[std::make_pair(i, movie)];

		}

		//derived_norm_v = 0;
		derived_regularized_loss_v = 0;

		//Initialize V variables for gradient descent
		double derived_regularized_gradient_v = 0;
		double derived_regularized_gradient_descent_v = 0;

		derived_norm_V = derived_v_getter(n, K, lambda, V, movies);

		for (int j : movies) {
			//for (int k = 0; k < K; k++) {

				//
			//}
			//derived_norm_v = derived_v_getter(derived_norm_v, lambda, V[j][t]);
			//derived_regularized_gradient_descent_v = gradient_descent_v_utility(eta, ratings, derived_norm_v, V_transposed);

		}

		std::cout << "Finished iteration " << t << endl;
	}

	std::cout << "Finish Gradient Descent" << std::endl;

	// calculate the mean absolute error
	double mae = 0;
	double mae_random = 0; // mean absolute error if we were to guess 3 for every rating

	for (auto& x : test_set) {
		int i = x.first.first;
		int j = x.first.second;
		double r = x.second;
		double prediction = dot_product(U[i], V[j]);
		if (prediction > 5) {
			prediction = 5;
		}
		else if (prediction < 1) {
			prediction = 1;
		}
		mae += abs(dot_product(U[i], V[j]) - r);
		mae_random += abs(3 - r);
	}

	mae = static_cast<double>(mae / test_set.size());
	mae_random = static_cast<double>(mae_random / test_set.size());
	std::cout << "Mean Absolute Error: " << mae << std::endl;
	std::cout << "Mean Absolute Error Random Guess: " << mae_random << std::endl;


	//Stochastic

	//relevant U reset
	//derived_norm_u = 0;
	derived_regularized_loss_u = 0;

	//initialize U variables for stochastic gradient descent
	double derived_regularized_stochastic_gradient_u = 0;
	double derived_regularized_stochastic_gradient_descent_u = 0;

	//relevant V reset
	//derived_norm_v = 0;
	derived_regularized_loss_v = 0;

	//initialize V variables for stochastic gradient descent
	double derived_regularized_stochastic_gradient_v = 0;
	double derived_regularized_stochastic_gradient_descent_v = 0;


	return 0;
}

