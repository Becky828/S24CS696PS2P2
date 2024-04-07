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


//function which performs collaborative filtering stochastic gradient descent
std::vector<std::vector<std::vector<double>>> cf_stochastic_gradient_descent_finder(std::map<std::pair<int, int>, double> test_set, int n_iterations, double eta, double lambda, double decay, std::set<int> users, std::set<int>  movies, std::map<std::pair<int, int>,
	double> ratings, double U_dot_V_transposed, double V_dot_U, std::map<int, std::set<int>> users_movies, std::map<int, std::set<int>> movies_users, int m, int n, int K, std::vector<std::vector<double>> U, std::vector<std::vector<double>> V) {

	//initializes the updated U and V
	std::vector<std::vector<std::vector<double>>> updated_U_V;

	// initializes mt19937 
	// object for finding random numbers 
	std::mt19937 mt(time(nullptr));

	//performs 0 to n_iterations of collaboarative filtering stochastic gradient descent incrementing by 1
	for (int t = 0; t < n_iterations; t++) {
		eta = eta * decay; // decay the learning rate over time

		//initializes the stochastic gradient base for U and V
		std::vector<std::vector<double>>cf_stochastic_gradient_base_U(n, std::vector<double>(K, 0));
		std::vector<std::vector<double>>cf_stochastic_gradient_base_V(n, std::vector<double>(K, 0));

		//randomly iterates to a rating
		auto it = ratings.begin();
		int random = mt() % ratings.size();
		std::advance(it, random);

		//stores the key of the randomly selected rating
		auto random_key = it->first;

		//stores the randomly selected user
		int i = random_key.first;
		int current_user = i;

		//stores the randomly selected movie
		int j = random_key.second;
		int current_movie = j;

		//stores the randomly selected rating value
		double current_rating = it->second;

		//iterates through the set of users by an increment of 1
		for (int a : users) {

			//iterates through the columns of U by an increment of 1
			for (int k = 0; k < K; k++) {

				//finds the dot product of U and V transposed
				U_dot_V_transposed = dot_product(U[a], V[j]);
				//U_dot_V_transposed = dot_product(U[a], V[j]);

				//finds the rating difference
				double rating_difference = U_dot_V_transposed - current_rating;

				//finds the base gradient for U
				cf_stochastic_gradient_base_U[a][k] = (rating_difference * V[j][k]);

				//performs the base gradient descent for U
				U[a][k] = U[a][k] - eta * cf_stochastic_gradient_base_U[a][k];

				//performs the regularization gradient descent for U
				U[a][k] = U[a][k] - eta * (2 * lambda * U[a][k]);
			}
		}

		//iterates through the set of movies by an increment of 1
		for (int a : movies) {

			//iterates through the columns of V by an increment of 1
			for (int k = 0; k < K; k++) {

				//finds the dot product of U and V transposed
				U_dot_V_transposed = dot_product(U[i], V[a]);

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

		//prints the current iteration
		std::cout << "Finished iteration " << t << endl;

		//finds the current mean absolute error
		mae_finder(test_set, U, V);
	}

	//prints that the stochastic gradient descent is finished
	std::cout << "Finished Collaboarative Filtering Stochastic Gradient Descent" << std::endl;

	//stores the updated U and V
	updated_U_V.push_back(U);
	updated_U_V.push_back(V);

	//returns the updated U and V
	return updated_U_V;
}



//function which performs collaborative filtering batch gradient descent
std::vector<std::vector<std::vector<double>>> cf_batch_gradient_descent_finder(int n_iterations, double eta, double lambda, double decay, std::set<int> users, std::set<int>  movies, std::map<std::pair<int, int>,
	double> ratings, double U_dot_V_transposed, double V_dot_U, std::map<int, std::set<int>> users_movies, std::map<int, std::set<int>> movies_users, int m, int n, int K, std::vector<std::vector<double>> U, std::vector<std::vector<double>> V) {

	//initializes the updated U and V
	std::vector<std::vector<std::vector<double>>> updated_U_V;

	//performs 0 to n_iterations of collaborative filtering gradient descent incrementing by 1
	for (int t = 0; t < n_iterations; t++) {
		eta = eta * decay; // decay the learning rate over time

		// implement gradient descent here:
		// you may want to use for (int i : users) and for (int j : movies) 
		// to iterate over all users and movies instead of for (int i = 0; i < m; i++) and for (int j = 0; j < n; j++)
		// to avoid iterating over users and movies that are not in the training set

		// you may also want to use the dot_product function to calculate the dot product of U[i] and V[j]
		// and the derived_u_getter and derived_v_getter functions to calculate the sum of the derived U and V values
		// you can also use the lambda, eta, and decay variables


		//iterates through the set of users
		for (int i : users) {

			//stores the current user
			int current_user = i;

			//stores the current user's movie set
			std::set<int> current_user_movie_set = users_movies[current_user];

			//initializes the base gradient for U. This ensures that the base gradient for U is set to 0 for each user
			std::vector<std::vector<double>>cf_batch_gradient_base_U(n, std::vector<double>(K, 0));

			//iterates through all the columns of U by an increment of 1
			for (int k = 0; k < K; k++) {

				//performs the summation of the base gradient for all samples relating to U
				for (int j : current_user_movie_set) {
					int current_movie = j;

					//finds the dot product of U and V transposed
					U_dot_V_transposed = dot_product(U[i], V[j]);

					//finds the current rating
					double current_rating = ratings.at(std::make_pair(current_user, current_movie));

					//finds the current rating difference
					double rating_difference = U_dot_V_transposed - current_rating;

					//updates the base gradient for U 
					// by adding the product of the difference between the dot product of U and V transposed and the current rating 
					// and the current element of V 
					// to the current element of the base gradient for U
					cf_batch_gradient_base_U[i][k] = cf_batch_gradient_base_U[i][k] + (rating_difference)*V[j][k];
				}

				//performs the base gradient descent for U
				U[i][k] = U[i][k] - eta * (cf_batch_gradient_base_U[i][k]);

				//performs the regularization gradient descent for U
				U[i][k] = U[i][k] - eta * (2 * lambda * U[i][k]);
			}
		}

		//iterates through the set of movies by an increment of 1
		for (int j : movies) {

			//stores the current movie
			int current_movie = j;

			//stores the current movie's user set
			std::set<int> current_movie_user_set = movies_users[j];

			//initializes the base gradient for U. This ensures that the base gradient for U is set to 0 for each movie
			std::vector<std::vector<double>> cf_batch_gradient_base_V(m, std::vector<double>(K, 0));

			//iterates through all the columns of V by an increment of 1
			for (int k = 0; k < K; k++) {
				for (int i : current_movie_user_set) {

					//finds the dot product of U and V transposed
					U_dot_V_transposed = dot_product(U[i], V[j]);

					//stores the current user
					int current_user = i;

					//finds the current rating
					double current_rating = ratings.at(std::make_pair(current_user, current_movie));

					//finds the current rating difference
					double rating_difference = U_dot_V_transposed - current_rating;

					//updates the base gradient for V 
					// by adding the product of the difference between the dot product of U and V transposed and the current rating 
					// and the current element of U
					// to the current element of the base gradient for V
					cf_batch_gradient_base_V[j][k] = cf_batch_gradient_base_V[j][k] + (rating_difference)*U[i][k];
				}

				//performs the base gradient descent for V
				V[j][k] = V[j][k] - eta * (cf_batch_gradient_base_V[j][k]);

				//performs the regularization gradient descent for V
				V[j][k] = V[j][k] - eta * (2 * lambda * V[j][k]);
			}
		}

		//prints the current iteration
		std::cout << "Finished iteration " << t << endl;
	}

	//prints that the gradient descent is finished
	std::cout << "Finished Collaboarative Filtering Batch Gradient Descent" << std::endl;

	//stores the updated U and V
	updated_U_V.push_back(U);
	updated_U_V.push_back(V);

	//returns the updated U and V
	return updated_U_V;
}

//function which performs collborative filtering  mini-batch gradient descent
std::vector<std::vector<std::vector<double>>> cf_mini_batch_gradient_descent_finder(int batch_size, std::map<std::pair<int, int>, double> test_set, int n_iterations, double eta, double lambda, double decay, std::set<int> users, std::set<int>  movies, std::map<std::pair<int, int>,
	double> ratings, double U_dot_V_transposed, double V_dot_U, std::map<int, std::set<int>> users_movies, std::map<int, std::set<int>> movies_users, int m, int n, int K, std::vector<std::vector<double>> U, std::vector<std::vector<double>> V) {


	//for debugging
	//batch_size = 100;

	//initializes the updated U and V
	std::vector<std::vector<std::vector<double>>> updated_U_V;

	// initializes mt19937 
	// object for finding random numbers  
	std::mt19937 mt(time(nullptr));


	//performs 0 to n_iterations of mini-batch gradient descent incrementing by 1
	for (int t = 0; t < n_iterations; t++) {
		eta = eta * decay; // decay the learning rate over time

		//initializes batch as a subset of ratings keys
		//std::set<std::pair<int, int>> ratings_keys_batch;
		std::map<std::pair<int, int>, double> ratings_batch;

		//iterates through the batch size by an increment of 1 to find a subset of ratings
		for (int a = 0; a < batch_size; a++) {

			//randomly iterates to a rating
			auto it = ratings.begin();
			int random = mt() % ratings.size();
			std::advance(it, random);

			//stores the key of the randomly selected rating			
			ratings_batch.insert(*it);

		}

		//iterates through the set of users by an increment of 1. This provides the index required for iterating through the rows of U
		//for (int a : users) {
			//int current_user = i;
			//std::set<int> current_user_movie_set = users_movies[current_user];
		int a = -1;
		for (auto it : ratings_batch) {
			++a;
			std::vector<std::vector<double>>cf_mini_batch_gradient_base_U(n, std::vector<double>(K, 0));
			//iterates through all the columns of U by an increment of 1
			for (int k = 0; k < K; k++) {

				//for (auto it : ratings_batch) {
					//int j = it.first.second;
				int i = it.first.first;

				//stores the current user
				int current_user = i;

				//stores the current user's movie set
				std::set<int> current_user_movie_set = users_movies[current_user];


				//performs the summation of the base gradient for all samples relating to U
				for (int j : current_user_movie_set) {
					int current_movie = j;
					auto candidate = std::make_pair(current_user, current_movie);
					if (ratings_batch.contains(candidate)) {
						//int current_movie = j;

						//finds the dot product of U and V transposed
						U_dot_V_transposed = dot_product(U[i], V[j]);

						//finds the current rating
						double current_rating = ratings.at(candidate);

						//finds the current rating difference
						double rating_difference = U_dot_V_transposed - current_rating;

						//updates the base gradient for U 
						// by adding the product of the difference between the dot product of U and V transposed and the current rating 
						// and the current element of V 
						// to the current element of the base gradient for U
						cf_mini_batch_gradient_base_U[i][k] = cf_mini_batch_gradient_base_U[i][k] + (rating_difference)*V[j][k];
					}
				}

				//performs the base gradient descent for U
				U[a][k] = U[a][k] - eta * (cf_mini_batch_gradient_base_U[a][k]);

				//performs the regularization gradient descent for U
				U[a][k] = U[a][k] - eta * (2 * lambda * U[a][k]);
			}


		}
		//}
		a = -1;
		//performs the summation of the base gradient for a subset of samples relating to V
		for (auto it : ratings_batch) {
			++a;
			//iterates through the set of movies by an increment of 1. This provides the index required for iterating through the rows of V
					//for (int a : movies) {
						//int current_movie = j;
			std::vector<std::vector<double>>cf_mini_batch_gradient_base_V(n, std::vector<double>(K, 0));
			//std::set<int> current_movie_user_set = movies_users[j];

			//iterates through all the columns of V by an increment of 1
			for (int k = 0; k < K; k++) {

				////performs the summation of the base gradient for a subset of samples relating to V
				//for (auto it : ratings_batch) {
				int j = it.first.second;

				//stores the current user
				int current_movie = j;

				//stores the current user's movie set
				std::set<int> current_movie_user_set = users_movies[current_movie];



				for (int i : current_movie_user_set) {

					int current_user = i;
					auto candidate = std::make_pair(current_user, current_movie);
					if (ratings_batch.contains(candidate)) {						//int j = it.first.second;
						//int current_movie = j;

						//finds the dot product of U and V transposed
						U_dot_V_transposed = dot_product(U[i], V[j]);

						//finds the current rating
						double current_rating = ratings.at(candidate);

						//finds the current rating difference
						double rating_difference = U_dot_V_transposed - current_rating;

						//updates the base gradient for V 
						// by adding the product of the difference between the dot product of U and V transposed and the current rating 
						// and the current element of U 
						// to the current element of the base gradient for U
						cf_mini_batch_gradient_base_V[i][k] = cf_mini_batch_gradient_base_V[i][k] + (rating_difference)*U[i][k];
					}
				}

				//performs the base gradient descent for V
				V[a][k] = V[a][k] - eta * (cf_mini_batch_gradient_base_V[a][k]);

				//performs the regularization gradient descent for V
				V[a][k] = V[a][k] - eta * (2 * lambda * V[a][k]);
			}
		}


		//prints the current iteration
		std::cout << "Finished iteration " << t << endl;
		mae_finder(test_set, U, V);
	}

	//prints that the gradient descent is finished
	std::cout << "Finished Collaboarative Filtering Mini-Batch Gradient Descent" << std::endl;

	//stores the updated U and V
	updated_U_V.push_back(U);
	updated_U_V.push_back(V);

	//returns the updated U and V
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
	int batch_size = 0;
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
	int n_iterations_copy = n_iterations;
	int n_iterations_double = 2 * n_iterations;
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

	//	batch_size = ratings.size() * 0.010;

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
	eta = 1000 * eta;
	//lambda = lambda / 90;
	lambda = lambda_10_times_up;
	n_iterations = 5 * n_iterations;
	//n_iterations = m;
	//epochs = 100;
	//lambda = lambda_copy;

	std::cout << "\n" << "\n" << "Stochastic Gradient Descent:" << std::endl;
	updated_U_V = cf_stochastic_gradient_descent_finder(test_set, n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);
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

	//n_iterations = m;
	//resetting U_dot_V_transposed and V_dot_U
	U_dot_V_transposed = 0;
	V_dot_U = 0;
	eta = eta_copy;
	lambda = lambda_copy;
	//eta = eta_10_times_up;
	batch_size = 200;
	n_iterations = 6 * n_iterations_copy;
	std::cout << "\n" << "\n" << "Mini-Batch Gradient Descent:" << std::endl;
	updated_U_V = cf_mini_batch_gradient_descent_finder(batch_size, test_set, n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);
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

	n_iterations = n_iterations_copy;
	eta = eta_copy;
	lambda = lambda_copy;

	// 1 of 5
	// gradient descent found with given hyperparameters
	std::cout << "Gradient Descent:" << std::endl;
	std::cout << "1 of 5:" << std::endl;
	std::cout << "Given Hyperparameters" << std::endl;
	updated_U_V = cf_batch_gradient_descent_finder(n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);

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
	n_iterations = n_iterations_double;

	//resets the eta to the original value
	eta = eta_copy;

	// 2 of 5
	//gradient descent found with the doubled number of iterations
	std::cout << "2 of 5:" << std::endl;
	std::cout << "Doubled Number of Iterations" << std::endl;
	updated_U_V = cf_batch_gradient_descent_finder(n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);

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
	updated_U_V = cf_batch_gradient_descent_finder(n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);

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
	updated_U_V = cf_batch_gradient_descent_finder(n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);

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
	updated_U_V = cf_batch_gradient_descent_finder(n_iterations, eta, lambda, decay, users, movies, ratings, U_dot_V_transposed, V_dot_U, users_movies, movies_users, m, n, K, U, V);

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