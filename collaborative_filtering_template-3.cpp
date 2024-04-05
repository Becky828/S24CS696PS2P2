#include <iostream>
#include <fstream>
#include <istream>
#include <random>
#include <set>
#include <map>
#include <vector>
#include <sstream>
#include <random>

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


void mae_finder(std::map<std::pair<int, int>, double> test_set, std::vector<std::vector<double>> U, std::vector<std::vector<double>> V)
{
	//Put MAE here


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
}


//Put V transposer here
std::vector<std::vector<double>>  v_transposer(int K, std::vector<std::vector<double>> V, std::set<int> movies, int n) {
	//std::vector<std::vector<double>>  v_transposer(int K, std::vector<double>& v1, std::set<int> movies, int n) {

		//int n = V.size();
		//std::vector<std::vector<double>> V_transposed(n, std::vector<double>(K, 0));
		//    vector<vector<int> > trans_vec(b[0].size(), vector<int>());

	std::vector<std::vector<double>> V_transposed(V[0].size(), std::vector<double>(n, 0));
	for (int j : movies) {
		for (int k = 0; k < K; k++) {
			V_transposed[k][j] = V[j][k];
		}

	}

	return V_transposed;
}

//Put custom functions here
std::vector<std::vector<double>> derived_u_getter(int m, int K, double lambda, std::vector<std::vector<double>> U, std::set<int> users) {
	std::vector<std::vector<double>> regularized_U(m, std::vector<double>(K, 0));
	for (int i : users) {
		for (int k = 0; k < K; k++) {
			regularized_U[i][k] = 2 * lambda * U[i][k];
		}
	}
	return regularized_U;
}

std::vector<std::vector<double>> derived_v_getter(int n, int K, double lambda, std::vector<std::vector<double>> V, std::set<int> movies) {
	std::vector<std::vector<double>> regularized_V(n, std::vector<double>(K, 0));
	for (int j : movies) {
		for (int k = 0; k < K; k++) {
			regularized_V[j][k] = 2 * lambda * V[j][k];
		}
	}
	return regularized_V;
}



//
std::vector<std::vector<std::vector<double>>> gradient_descent_finder(int n_iterations, double eta, double lambda, double decay, std::set<int> users, std::set<int>  movies, std::map<std::pair<int, int>,
	double> ratings, double U_dot_V_transposed, double V_dot_U, std::map<int, std::set<int>> users_movies, std::map<int, std::set<int>> movies_users, int m, int n, int K, std::vector<std::vector<double>> U, std::vector<std::vector<double>> V) {

	//Put gradient descent here

	std::vector<std::vector<std::vector<double>>> updated_U_V;

	for (int t = 0; t < n_iterations; t++) {
		eta = eta * decay; // decay the learning rate over time


		// implement gradient descent here:
		// you may want to use for (int i : users) and for (int j : movies) 
		// to iterate over all users and movies instead of for (int i = 0; i < m; i++) and for (int j = 0; j < n; j++)
		// to avoid iterating over users and movies that are not in the training set

		// you may also want to use the dot_product function to calculate the dot product of U[i] and V[j]
		// and the derived_u_getter and derived_v_getter functions to calculate the sum of the derived U and V values
		// you can also use the lambda, eta, and decay variables

		for (int i : users) {
			int current_user = i;
			std::set<int> current_user_movie_set = users_movies[current_user];
			std::vector<std::vector<double>>cf_gradient_base_U(n, std::vector<double>(K, 0));

			for (int k = 0; k < K; k++) {

				for (int j : current_user_movie_set) {
					int current_movie = j;
					U_dot_V_transposed = dot_product(U[i], V[j]);
					double current_rating = ratings.at(std::make_pair(current_user, current_movie));
					//double current_rating = ratings[std::make_pair(current_user, current_movie)];
					cf_gradient_base_U[i][k] = cf_gradient_base_U[i][k] + (U_dot_V_transposed - current_rating) * V[j][k];
				}

				//performs the base gradient descent for U
				U[i][k] = U[i][k] - eta * (cf_gradient_base_U[i][k]);

				//performs the regularization gradient descent for U
				U[i][k] = U[i][k] - eta * (2 * lambda * U[i][k]);
			}
		}

		for (int j : movies) {
			int current_movie = j;
			std::vector<std::vector<double>> cf_gradient_base_V(m, std::vector<double>(K, 0));
			std::set<int> current_movie_user_set = movies_users[j];

			for (int k = 0; k < K; k++) {
				for (int i : current_movie_user_set) {
					//V_dot_U = dot_product(V[j], U[i]);
					U_dot_V_transposed = dot_product(U[i], V[j]);
					int current_user = i;
					double current_rating = ratings.at(std::make_pair(current_user, current_movie));
					//double current_rating = ratings[std::make_pair(current_user, current_movie)];
					//cf_gradient_base_V[j][k] = cf_gradient_base_V[j][k] + (V_dot_U - current_rating) * U[i][k];
					cf_gradient_base_V[j][k] = cf_gradient_base_V[j][k] + (U_dot_V_transposed - current_rating) * U[i][k];
				}

				//performs the base gradient descent for V
				V[j][k] = V[j][k] - eta * (cf_gradient_base_V[j][k]);

				//performs the regularization gradient descent for V
				V[j][k] = V[j][k] - eta * (2 * lambda * V[j][k]);
			}
		}

		std::cout << "Finished iteration " << t << endl;
	}

	std::cout << "Finish Gradient Descent" << std::endl;

	//stores the updated U and V
	updated_U_V.push_back(U);
	updated_U_V.push_back(V);
	return updated_U_V;
}


//Put stochastic u gradient descent here
std::vector<std::vector<std::vector<double>>> stochastic_gradient_descent_finder_1(std::map<std::pair<int, int>, double> test_set, int n_iterations, double eta, double lambda, double decay, std::set<int> users, std::set<int>  movies, std::map<std::pair<int, int>,
	double> ratings, double U_dot_V_transposed, double V_dot_U, std::map<int, std::set<int>> users_movies, std::map<int, std::set<int>> movies_users, int m, int n, int K, std::vector<std::vector<double>> U, std::vector<std::vector<double>> V) {

	//Put gradient descent here

	std::vector<std::vector<std::vector<double>>> updated_U_V;

	/*std::set<int> previous_users;
	std::set<int> previous_movies;*/

	std::set<int> avaialble_users = users;
	std::set<int> available_movies = movies;

	std::set<int> previous_users;
	std::set<int> previous_movies;

	int previous_user = 0;
	int previous_movie = 0;

	// Initializing mt19937 
  // object 
	std::mt19937 mt(time(nullptr));

	for (int t = 0; t < n_iterations; t++) {
		eta = eta * decay; // decay the learning rate over time

		//initialize the stochastic gradient base for U and V
		std::vector<std::vector<double>>cf_stochastic_gradient_base_U(n, std::vector<double>(K, 0));
		std::vector<std::vector<double>>cf_stochastic_gradient_base_V(n, std::vector<double>(K, 0));

		//an issue with the stochastic gradient descent is that it is not updating the U and V values correctly after a few iterations

		//randomly iterate to a rating
		auto it = ratings.begin();
		//std::advance(it, rand() % ratings.size());
		//int next = mt19937();
		//auto result = next % ratings.size();
		//int random = generate_uniform_random_number();
		
		//std::advance(it, mt() % ratings.size());
		int random = mt() % ratings.size();
		std::advance(it, random);
		//store the key of the randomly selected rating
		auto random_key = it->first;

		//store the randomly selected user
		int i = random_key.first;
		int current_user = i;

		//store the randomly selected movie
		int j = random_key.second;
		int current_movie = j;

		//store the randomly selected rating value
		double current_rating = it->second;

		//take the dot product of U and V transposed
		//U_dot_V_transposed = dot_product(U[i], V[j]);

		//find the rating difference
		//double rating_difference = U_dot_V_transposed - current_rating;

		for (int a : users) {

			for (int k = 0; k < K; k++) {

				//finds the dot product of U and V transposed
				U_dot_V_transposed = dot_product(U[a], V[j]);

				//finds the current rating
				//double current_rating = ratings.at(std::make_pair(a, j));

				//finds the rating difference
				double rating_difference = U_dot_V_transposed - current_rating;

				//finds the base gradient for U
				cf_stochastic_gradient_base_U[a][k] = (rating_difference * V[j][k]);

				//performs the base gradient descent for U
				U[a][k] = U[a][k] - eta * cf_stochastic_gradient_base_U[a][k];

				//performs the regularization gradient descent for U
				U[a][k] = U[a][k] - eta * (2 * lambda * U[a][k]);
			}

			//U[a] = U[a] - eta * cf_stochastic_gradient_base_U;
		}

		for (int a : movies) {

			for (int k = 0; k < K; k++) {

				//finds the dot product of U and V transposed
				U_dot_V_transposed = dot_product(U[i], V[a]);

				//finds the current rating
				//double current_rating = ratings.at(std::make_pair(i, j));

				//finds the rating difference
				double rating_difference = U_dot_V_transposed - current_rating;

				//finds the base gradient for V
				cf_stochastic_gradient_base_V[a][k] = (rating_difference * U[i][k]);
				//performs the base gradient descent for V
				V[a][k] = V[a][k] - eta * (cf_stochastic_gradient_base_V[a][k]);

				//performs the regularization gradient descent for V
				V[a][k] = V[a][k] - eta * (2 * lambda * V[a][k]);
			}
		}

		std::cout << "Finished iteration " << t << endl;
		mae_finder(test_set, U, V);
	}

	std::cout << "Finish Stochastic Gradient Descent" << std::endl;

	//stores the updated U and V
	updated_U_V.push_back(U);
	updated_U_V.push_back(V);
	return updated_U_V;
}


std::vector<std::vector<std::vector<double>>> stochastic_gradient_descent_finder_2(std::map<std::pair<int, int>, double> test_set, int n_iterations, double eta, double lambda, double decay, std::set<int> users, std::set<int>  movies, std::map<std::pair<int, int>,
	double> ratings, double U_dot_V_transposed, double V_dot_U, std::map<int, std::set<int>> users_movies, std::map<int, std::set<int>> movies_users, int m, int n, int K, std::vector<std::vector<double>> U, std::vector<std::vector<double>> V) {
	//Follows method 2 of the stochastic gradient descent as detailed in the stochastic gradient descent slides
	//Put gradient descent here

	std::vector<std::vector<std::vector<double>>> updated_U_V;

	/*std::set<int> previous_users;
	std::set<int> previous_movies;*/

	std::set<int> avaialble_users = users;
	std::set<int> available_movies = movies;

	std::set<int> previous_users;
	std::set<int> previous_movies;

	int previous_user = 0;
	int previous_movie = 0;

	// Initializing mt19937 
  // object 
	std::mt19937 mt(time(nullptr));

	for (int t = 0; t < n_iterations; t++) {
		eta = eta * decay; // decay the learning rate over time

		//initialize the stochastic gradient base for U and V
		std::vector<std::vector<double>>cf_stochastic_gradient_base_U(n, std::vector<double>(K, 0));
		std::vector<std::vector<double>>cf_stochastic_gradient_base_V(n, std::vector<double>(K, 0));

		//an issue with the stochastic gradient descent is that it is not updating the U and V values correctly after a few iterations

		//randomly iterate to a rating
		auto it = ratings.begin();
		//std::advance(it, rand() % ratings.size());
		//int next = mt19937();
		//auto result = next % ratings.size();
		//int random = generate_uniform_random_number();

		//std::advance(it, mt() % ratings.size());
		int random = mt() % ratings.size();
		std::advance(it, random);
		//store the key of the randomly selected rating
		auto random_key = it->first;

		//store the randomly selected user
		int i = random_key.first;
		int current_user = i;

		//store the randomly selected movie
		int j = random_key.second;
		int current_movie = j;

		//store the randomly selected rating value
		double current_rating = it->second;

		//take the dot product of U and V transposed
		//U_dot_V_transposed = dot_product(U[i], V[j]);

		//find the rating difference
		//double rating_difference = U_dot_V_transposed - current_rating;

		for (int a : users) {

			for (int k = 0; k < K; k++) {

				//finds the dot product of U and V transposed
				U_dot_V_transposed = dot_product(U[a], V[j]);

				//finds the current rating
				//double current_rating = ratings.at(std::make_pair(a, j));

				//finds the rating difference
				double rating_difference = U_dot_V_transposed - current_rating;

				//finds the base gradient for U
				cf_stochastic_gradient_base_U[a][k] = (rating_difference * V[j][k]);

				//performs the base gradient descent for U
				U[a][k] = U[a][k] - eta * cf_stochastic_gradient_base_U[a][k];

				//performs the regularization gradient descent for U
				U[a][k] = U[a][k] - eta * (2 * lambda * U[a][k]);
			}

			//U[a] = U[a] - eta * cf_stochastic_gradient_base_U;
		}

		for (int a : movies) {

			for (int k = 0; k < K; k++) {

				//finds the dot product of U and V transposed
				U_dot_V_transposed = dot_product(U[i], V[a]);

				//finds the current rating
				//double current_rating = ratings.at(std::make_pair(i, j));

				//finds the rating difference
				double rating_difference = U_dot_V_transposed - current_rating;

				//finds the base gradient for V
				cf_stochastic_gradient_base_V[a][k] = (rating_difference * U[i][k]);
				//performs the base gradient descent for V
				V[a][k] = V[a][k] - eta * (cf_stochastic_gradient_base_V[a][k]);

				//performs the regularization gradient descent for V
				V[a][k] = V[a][k] - eta * (2 * lambda * V[a][k]);
			}
		}

		std::cout << "Finished iteration " << t << endl;
		mae_finder(test_set, U, V);
	}

	std::cout << "Finish Stochastic Gradient Descent" << std::endl;

	//stores the updated U and V
	updated_U_V.push_back(U);
	updated_U_V.push_back(V);
	return updated_U_V;
}

int main() {

	//for quick debugging
	//std::ifstream file("very_abridged_Dataset.csv");

	//for first part of p2
	std::ifstream file("Dataset.csv");
	// std::ifstream file("Movie_Id_Titles.csv");    

	//for second part of p2
	//std::ifstream file("ratings.csv");

	std::string line;
	std::map<std::pair<int, int>, double> ratings;
	std::map<std::pair<int, int>, double> test_set;
	std::map<int, std::set<int>> users_movies;
	std::map<int, std::set<int>> movies_users;
	std::set<int> users;
	std::set<int> movies;

	//Full Dataset
	int K = 15; // number of latent dimensions
	int m = 2000; // upper bound for number of users
	int n = 2000; // upper bound number of movies

	//Abirdged Dataset
	//int K = 15; // number of latent dimensions
	//int m = 500; // upper bound for number of users
	//int n = 500; // upper bound number of movies

	double test_set_size = 0.1; // percentage of the data will be used for testing
	double lambda = 1e-3; // regularization parameter
	double lambda_copy = lambda;
	double lambda_10_times_up = lambda * 10;
	double lambda_10_times_down = lambda / 10;
	double eta = 1e-4; // learning rate
	double eta_copy = eta;
	double eta_10_times_up = eta * 10;
	double eta_10_times_down = eta / 10;
	double decay = 0.9; // decay rate
	int n_iterations = 35; // number of iterations for the gradient descent
	int n_interations_copy = n_iterations;
	int n_interations_double = 2 * n_iterations;
	double U_dot_V_transposed = 0;
	double V_dot_U = 0;

	//int epochs = 4;

	std::vector<std::vector<std::vector<double>>> updated_U_V;


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
				//double current_rating_a = ratings[std::make_pair(user, movie)];
				users_movies[user].insert(movie); // add movie to user's list of movies
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

	// initialize copiesa of U and V for U and V reset
	std::vector<std::vector<double>> copy_U(m, std::vector<double>(K, 0));
	std::vector<std::vector<double>> copy_V(n, std::vector<double>(K, 0));

	//initialize the partial derivatives for U and V
	std::vector<std::vector<double>> derived_norm_U(m, std::vector<double>(K, 0));
	std::vector<std::vector<double>> derived_norm_V(n, std::vector<double>(K, 0));

	//Using variables for debuggine purposes

	//initialize the difference in ratings for U and V 
	std::vector<std::vector<double>> ratings_difference_U_product(m, std::vector<double>(K, 0));
	std::vector<std::vector<double>> ratings_difference_V_product(n, std::vector<double>(K, 0));

	//initialize the gradient for U and V base
	//std::vector<std::vector<double>> cf_gradient_base_U(m, std::vector<double>(K, 0));
	//std::vector<std::vector<double>> cf_gradient_base_V(n, std::vector<double>(K, 0));

	//initialize the gradient regularization for U and V
	std::vector<std::vector<double>> cf_gradient_regularization_U(m, std::vector<double>(K, 0));
	std::vector<std::vector<double>> cf_gradient_regularization_V(n, std::vector<double>(K, 0));

	////initialize the difference for U and V base
	//std::vector<std::vector<double>> U_difference_base(m, std::vector<double>(K, 0));
	//std::vector<std::vector<double>> V_difference_base(n, std::vector<double>(K, 0));

	////initialize the difference for U and V regularization
	//std::vector<std::vector<double>> U_difference_regularization(m, std::vector<double>(K, 0));
	//std::vector<std::vector<double>> V_difference_regularization(n, std::vector<double>(K, 0));

	//initialize the gradient descent for U and V
	std::vector<std::vector<double>> cf_gradient_descent_U(m, std::vector<double>(K, 0));
	std::vector<std::vector<double>> cf_gradient_descent_V(n, std::vector<double>(K, 0));

	//initialize the stochastic gradient descent for U and V
	std::vector<std::vector<double>> cf_stochastic_gradient_descent_U(m, std::vector<double>(K, 0));
	std::vector<std::vector<double>> cf_stochastic_gradient_descent_V(n, std::vector<double>(K, 0));


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

	//initialize the copy of U and V
	copy_U = U;
	copy_V = V;


	//eta = eta_copy;
	//n_iterations = 100000;
	//n_iterations = 60;
	//n_iterations = 70;
	//n_iterations = m;
	////eta = eta * (1000 + 57.4);
	////eta = eta * 900;
	//eta = eta * 9000;

	//lambda = lambda / 90000;
	eta = 900 * eta;
	lambda = lambda / 90;
	//n_iterations = 5 * n_iterations;
	n_iterations = m;
	//epochs = 100;
	//lambda = lambda_copy;

	std::cout << "Stochastic Gradient Descent Method 1:" << std::endl;
	updated_U_V = stochastic_gradient_descent_finder_1(test_set, n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);
	//set U and V to the updated U and V
	U = updated_U_V[0];
	V = updated_U_V[1];
	updated_U_V.clear();
	mae_finder(test_set, U, V);


	//empty the updated_U_V vector
	updated_U_V.clear();

	//mae found for the given hyper parameters
	mae_finder(test_set, U, V);

	//resetting U and V
	U = copy_U;
	V = copy_V;

	//resetting U_dot_V_transposed and V_dot_U
	U_dot_V_transposed = 0;
	V_dot_U = 0;


	std::cout << "Stochastic Gradient Descent Method 2:" << std::endl;
	updated_U_V = stochastic_gradient_descent_finder_2(test_set, n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);
	//set U and V to the updated U and V
	U = updated_U_V[0];
	V = updated_U_V[1];
	updated_U_V.clear();
	mae_finder(test_set, U, V);


	//empty the updated_U_V vector
	updated_U_V.clear();

	//mae found for the given hyper parameters
	mae_finder(test_set, U, V);

	//resetting U and V
	U = copy_U;
	V = copy_V;

	//resetting U_dot_V_transposed and V_dot_U
	U_dot_V_transposed = 0;
	V_dot_U = 0;

	n_iterations = n_interations_copy;
	eta = eta_copy;
	lambda = lambda_copy;

	// 1 of 5
	// gradient descent found with given hyperparameters
	std::cout << "Gradient Descent:" << std::endl;
	std::cout << "1 of 5:" << std::endl;
	std::cout << "Given Hyperparameters" << std::endl;
	updated_U_V = gradient_descent_finder(n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);

	//set U and V to the updated U and V
	U = updated_U_V[0];
	V = updated_U_V[1];

	//empty the updated_U_V vector
	updated_U_V.clear();

	//mae found for the given hyper parameters
	mae_finder(test_set, U, V);

	//resetting U and V
	U = copy_U;
	V = copy_V;

	//resetting U_dot_V_transposed and V_dot_U
	U_dot_V_transposed = 0;
	V_dot_U = 0;

	//doubling the number of iterations appears to reduce the MAE by around 0.01
	n_iterations = n_interations_double;

	//resets the eta to the original value
	eta = eta_copy;

	// 2 of 5
	//gradient descent found with the doubled number of iterations
	std::cout << "2 of 5:" << std::endl;
	std::cout << "Doubled Number of Iterations" << std::endl;
	updated_U_V = gradient_descent_finder(n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);

	//set U and V to the updated U and V
	U = updated_U_V[0];
	V = updated_U_V[1];

	//empty the updated_U_V vector
	updated_U_V.clear();

	//mae found for the doubled number of iterations
	mae_finder(test_set, U, V);

	//resetting U and V
	U = copy_U;
	V = copy_V;

	//resetting U_dot_V_transposed and V_dot_U
	U_dot_V_transposed = 0;
	V_dot_U = 0;

	//muliplying the eta by 10 appears to reduce the MAE by around  0.15
	eta = eta_10_times_up;

	// 3 of 5
	//gradient descent found with the doubled number of iterations and eta times 10
	std::cout << "3 of 5:" << std::endl;
	std::cout << "Doubled Number of Iterations, eta times 10, and unchanged lambda" << std::endl;

	//gets updated V and U
	updated_U_V = gradient_descent_finder(n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);

	//set U and V to the updated U and V
	U = updated_U_V[0];
	V = updated_U_V[1];

	//empty the updated_U_V vector
	updated_U_V.clear();

	//mae found for the doubled number of iterations and eta times 10
	mae_finder(test_set, U, V);

	//resetting U and V
	U = copy_U;
	V = copy_V;

	//resettnng U_dot_V_transposed and V_dot_U
	U_dot_V_transposed = 0;
	V_dot_U = 0;

	//resets the eta to 10 times the original value
	eta = eta_10_times_up;

	//muliplying the lambda by 10 appears to slightly increase the MAE
	lambda = lambda_10_times_up;

	//4 of 5
	//gradient descent found with the doubled number of iterations, eta times 10, and lambda times 10
	std::cout << "4 of 5:" << std::endl;
	std::cout << "Doubled Number of Iterations, eta times 10, and lambda times 10" << std::endl;
	std::cout << "This provided the lowest found MAE." << std::endl;

	//gets updated V and U
	updated_U_V = gradient_descent_finder(n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);

	//set U and V to the updated U and V
	U = updated_U_V[0];
	V = updated_U_V[1];

	//empty the updated_U_V vector
	updated_U_V.clear();

	//mae found for the doubled number of iterations and eta times 10
	mae_finder(test_set, U, V);

	//resetting U and V
	U = copy_U;
	V = copy_V;

	//resettnng U_dot_V_transposed and V_dot_U
	U_dot_V_transposed = 0;
	V_dot_U = 0;

	//resets the eta to 10 times the original value
	eta = eta_10_times_up;

	//dividing the lambda by 10 appears to slightly increase the MAE
	lambda = lambda_10_times_down;

	//5 of 5
	//gradient descent found with the doubled number of iterations, eta times 10, and lambda times 10
	std::cout << "5 of 5:" << std::endl;
	std::cout << "Doubled Number of Iterations, eta times 10, and lambda divided by 10" << std::endl;

	//gets updated V and U
	updated_U_V = gradient_descent_finder(n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);

	//set U and V to the updated U and V
	U = updated_U_V[0];
	V = updated_U_V[1];

	//empty the updated_U_V vector
	updated_U_V.clear();

	//mae found for the doubled number of iterations and eta times 10
	mae_finder(test_set, U, V);


	//p2b
	//stochastic gradient descent

	/*eta = eta_copy;
	n_iterations = n_interations_double;
	lambda = lambda_copy;

	updated_U_V = stochastic_gradient_descent_finder(n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);*/


	return 0;

}