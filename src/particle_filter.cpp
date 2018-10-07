/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Modified on : 09/30, 2018
 *  	Author: kputtur
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	//DONE: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	// set std deviations for x, y  and theta
	auto std_x = std[0];
	auto std_y = std[1];
	auto std_theta = std[2];

	//This line creates a Gaussian distribution for x.
	normal_distribution<double> dist_x(x, std_x);

	//Create a normal distribution for y and theta
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	
	//Initialize the number of particles to 250
	this->num_particles = 250;
	this->particles.resize(this->num_particles);
	//set the x, y and theta values to random at the beginning
	for ( int i = 0; i < this->num_particles; ++i) {
		auto &p = this->particles[i];
		p.id = i;
		p.x  = dist_x(this->rand);
		p.y  = dist_y(this->rand);
		p.theta = dist_theta(this->rand);
		p.weight = 1;
	}
	this->weights = vector<double>(this->num_particles, 1.0);
	//finally mark it as initialized
	this->is_initialized = true;
	

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// DONE: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	// Set the standard deviations for x, y, and theta.
	auto std_x = std_pos[0];
	auto std_y = std_pos[1];
	auto std_theta = std_pos[2];


	//Create a normal distribution for x, y and theta.
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);	

	for (auto &p: this->particles) {
	       //move without error
	       if (fabs(yaw_rate) < 0.000001) {
	            p.x += velocity * delta_t * cos(p.theta);
     		    p.y += velocity * delta_t * sin(p.theta);
		    p.theta += 0.0;
		} else {
		    p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
		    p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
		    p.theta += yaw_rate * delta_t;
	       }

	     //randomized error
	     p.x += dist_x(this->rand);
	     p.y += dist_y(this->rand);
	     p.theta += dist_theta(this->rand);
	    
	    //normalize p.theta between 0 - (3.14 * 2) 
	     while(p.theta > M_PI * 2) {
		   p.theta -= M_PI * 2;
	     }

	    while (p.theta < 0 ) {
		  p.theta += M_PI * 2;  
	      }
       }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// DONE: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	if(observations.size() <= 0) {
		std::cout << "no observations were given " << std::endl;
	 }

	if(predicted.size() <= 0) {
		std::cout << "no predicted observations were given " << std::endl;
	}

	for (auto &obs : observations) {
		auto dist_squared_min = std::numeric_limits<const float>::infinity();
		obs.id = -2;
		for (auto const &pred : predicted) {
			auto dx = obs.x - pred.x;
			auto dy = obs.y - pred.y;
			auto dist_squared = dx * dx + dy * dy;

			if (dist_squared < dist_squared_min) {
				dist_squared_min = dist_squared;
				obs.id =  pred.id;
			}

		}

		if (obs.id == -2) {
			std::cout << "no nearest prediction found" << std::endl;
		 }
	 }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	
	// DONE: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	//keep a copy of observations
	auto obs_in_car_coords = observations;
	for(auto &obs : obs_in_car_coords) {
	   obs.id = -1;
   	}

	auto predict_in_car_coords = std::vector<LandmarkObs>();
	predict_in_car_coords.reserve(map_landmarks.landmark_list.size());
	
	auto sensor_range_squared = sensor_range * sensor_range;
	for(auto &p : this->particles) {
		predict_in_car_coords.clear();
		
		for(auto const &lm : map_landmarks.landmark_list) {
			auto dx = lm.x_f - p.x;
			auto dy = lm.y_f - p.y;
			auto distance_squared = dx * dx + dy * dy;

			if (distance_squared <= sensor_range_squared) {
				LandmarkObs pred_in_car_coordinates;
				pred_in_car_coordinates.id = lm.id_i;
				pred_in_car_coordinates.x = cos(p.theta) * (lm.x_f - p.x) + sin(p.theta) * (lm.y_f - p.y);
				pred_in_car_coordinates.y = -sin(p.theta) * (lm.x_f - p.x) + cos(p.theta) * (lm.y_f - p.y);
				predict_in_car_coords.push_back(pred_in_car_coordinates);

			} //closing if

		} //closing of for

		this->dataAssociation(predict_in_car_coords, obs_in_car_coords);

		p.weight = 1;

		for(auto const &obs : obs_in_car_coords) {
		   const auto pred_in_car_coordinates = std::find_if(
			 predict_in_car_coords.begin(),
			 predict_in_car_coords.end(),
			 [&obs](const LandmarkObs &x) { return x.id == obs.id; });

		
	  	if (pred_in_car_coordinates == predict_in_car_coords.end()) {
	     		//nothing found
			p.weight = 0;
			std::cout<< "error : Landmark not found for id=" << obs.id << std::endl;		
			break;
		}

	auto prob = normalize(obs.x, obs.y, pred_in_car_coordinates->x, pred_in_car_coordinates->y, std_landmark[0], std_landmark[1]);
	p.weight *= prob;
    } //closing for

  }

        //copying the weights from the particles into the array
	for (int i = 0; i < this->num_particles; ++i) {
		this->weights[i] = this->particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// DONE: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	//create a discrete distribution for particle index
	discrete_distribution<> dist(this->weights.begin(), this->weights.end());

	//resampling
	auto new_particles = std::vector<Particle>(this->num_particles);
	for (int i = 0; i < this->num_particles; ++i) {
		auto particle_index = dist(this->rand);
		auto &p = new_particles[i];
		p.id = i;
		p.x = this->particles[particle_index].x;
		p.y = this->particles[particle_index].y;
		p.theta = this->particles[particle_index].theta;
		p.weight = this->weights[particle_index];
	}
		this->particles = new_particles;
}
	

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
