#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <cassert>
#include <array>
#include <algorithm>

using real_t = float;
constexpr size_t NX = 512, NY = 512; //World Size
using grid_t = std::array<std::array<real_t, NX>, NY>;

class Sim_Configuration {
public:
    int iter = 10000;  // Number of iterations
    double dt = 0.05;       // Size of the integration time step
    real_t g = 9.80665;     // Gravitational acceleration
    real_t dx = 1;          // Integration step size in the horizontal direction
    real_t dy = 1;          // Integration step size in the vertical direction
    int data_period = 10000;// how often to save coordinate to file
    int numgangs = 1; 
    std::string filename = "sw_output.data";   // name of the output file with history

    Sim_Configuration(std::vector <std::string> argument){
        for (long unsigned int i = 1; i<argument.size() ; i += 2){
            std::string arg = argument[i];
            if(arg=="-h"){ // Write help
                std::cout << "./par --iter <number of iterations> --dt <time step>"
                          << " --g <gravitational const> --dx <x grid size> --dy <y grid size>"
                          << "--fperiod <iterations between each save> --out <name of output file> --numgangs <Number of gangs>\n";
                exit(0);
            } else if (i == argument.size() - 1)
                throw std::invalid_argument("The last argument (" + arg +") must have a value");
            else if(arg=="--iter"){
                if ((iter = std::stoi(argument[i+1])) < 0) 
                    throw std::invalid_argument("iter most be a positive integer (e.g. -iter 1000)");
            } else if(arg=="--dt"){
                if ((dt = std::stod(argument[i+1])) < 0) 
                    throw std::invalid_argument("dt most be a positive real number (e.g. -dt 0.05)");
            } else if(arg=="--g"){
                g = std::stod(argument[i+1]);
            } else if(arg=="--dx"){
                if ((dx = std::stod(argument[i+1])) < 0) 
                    throw std::invalid_argument("dx most be a positive real number (e.g. -dx 1)");
            } else if(arg=="--dy"){
                if ((dy = std::stod(argument[i+1])) < 0) 
                    throw std::invalid_argument("dy most be a positive real number (e.g. -dy 1)");
            } else if(arg=="--fperiod"){
                if ((data_period = std::stoi(argument[i+1])) < 0) 
                    throw std::invalid_argument("dy most be a positive integer (e.g. -fperiod 100)");
            } else if(arg=="--out"){
                filename = argument[i+1];
            } else if (arg=="--numgangs"){
                if ((numgangs = std::stoi(argument[i+1])) < 0) 
                    throw std::invalid_argument("numgangs most be a positive integer (e.g. -numgangs 1000)");
            } else{
                std::cout << "---> error: the argument type is not recognized \n";
            }
        }
    }
};

/** Representation of a water world including ghost lines, which is a "1-cell padding" of rows and columns
 *  around the world. These ghost lines is a technique to implement periodic boundary conditions. */
class Water {
public:
    grid_t u{}; // The speed in the horizontal direction.
    grid_t v{}; // The speed in the vertical direction.
    grid_t e{}; // The water elevation.
    Water() {
        for (size_t i = 1; i < NY - 1; ++i) 
        for (size_t j = 1; j < NX - 1; ++j) {
            real_t ii = 100.0 * (i - (NY - 2.0) / 2.0) / NY;
            real_t jj = 100.0 * (j - (NX - 2.0) / 2.0) / NX;
            e[i][j] = std::exp(-0.02 * (ii * ii + jj * jj));
        }
    }
};

void writeVectorVectorDoubleToBinary(const std::vector<grid_t>& data, const std::string& filename);


/* Write a history of the water heights to an ASCII file
 *
 * @param water_history  Vector of the all water worlds to write
 * @param filename       The output filename of the ASCII file
*/
void to_file(const std::vector<grid_t> &water_history, const std::string &filename){
    std::ofstream file(filename);
    file.write((const char*)(water_history.data()), sizeof(grid_t)*water_history.size());
}

/** One integration step
 *
 * @param w The water world to update.
 */
void integrate(Water &w, const real_t dt, const real_t dx, const real_t dy, const real_t g,const int numgangs) {
    
    #pragma acc parallel num_gangs(numgangs) present(w)
    {
        #pragma acc loop gang
        for (uint64_t j = 0; j < NX; ++j) {
            w.e[0][j]      = w.e[NY-2][j]; 
            w.e[NY-1][j]   = w.e[1][j];
            
            w.v[0][j]      = w.v[NY-2][j]; 
            w.v[NY-1][j]   = w.v[1][j];

        }
        #pragma acc loop gang
        for (uint64_t j = 0; j < NY; ++j) {

            w.e[j][0] = w.e[j][NX-2];
            w.e[j][NX-1] = w.e[j][1];
            
            w.u[j][0] = w.u[j][NX-2];
            w.u[j][NX-1] = w.u[j][1];
        }
        // #pragma acc wait
    }
    #pragma acc parallel num_gangs(numgangs) present(w) 
    {
        #pragma acc loop gang collapse(2)
        for (uint64_t i = 0; i < NY - 1; ++i) 
        for (uint64_t j = 0; j < NX - 1; ++j) {
            w.u[i][j] -= dt / dx * g * (w.e[i][j+1] - w.e[i][j]);
            w.v[i][j] -= dt / dy * g * (w.e[i + 1][j] - w.e[i][j]);
        }
    }

    #pragma acc parallel num_gangs(numgangs) present(w)
    {
        #pragma acc loop gang collapse(2) // gives hints about the grid // using tile
        for (uint64_t i = 1; i < NY - 1; ++i) 
        for (uint64_t j = 1; j < NX - 1; ++j) {
            w.e[i][j] -= dt / dx * (w.u[i][j] - w.u[i][j-1])
                    + dt / dy * (w.v[i][j] - w.v[i-1][j]);
        }
    }
}

/** Simulation of shallow water
 *
 * @param num_of_iterations  The number of time steps to simulate
 * @param size               The size of the water world excluding ghost lines
 * @param output_filename    The filename of the written water world history (HDF5 file)
 */
void simulate(const Sim_Configuration config) {
    Water water_world = Water();

    std::vector <grid_t> water_history;
    auto begin = std::chrono::steady_clock::now();
    #pragma acc data copy(water_world)
    {
        for (uint64_t t = 0; t < config.iter; ++t) {
            
            integrate(water_world, config.dt,  config.dx, config.dy, config.g, config.numgangs);
        
            if (t % config.data_period == 0) {
                #pragma acc update host(water_world.e)
                water_history.push_back(water_world.e);
            }
        }
    }
    auto end = std::chrono::steady_clock::now();

    to_file(water_history, config.filename);
    // writeVectorVectorDoubleToBinary(water_history,config.filename);


    std::cout << "checksum: " << std::accumulate(water_world.e.front().begin(), water_world.e.back().end(), 0.0) << std::endl;
    std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0 << " sec" << std::endl;
}

/** Main function that parses the command line and start the simulation */
int main(int argc, char **argv) {
    auto config = Sim_Configuration({argv, argv+argc});
    simulate(config);
    return 0;
}


void writeVectorVectorDoubleToBinary(const std::vector<grid_t>& data, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // Write the number of rows
    size_t numRows = data.size();
    outFile.write(reinterpret_cast<const char*>(&numRows), sizeof(numRows));

    for (const auto& row : data) {
        // Write the number of elements in each row
        size_t numElements = row.size();
        outFile.write(reinterpret_cast<const char*>(&numElements), sizeof(numElements));
        
        // Write the elements of the row
        outFile.write(reinterpret_cast<const char*>(row.data()), numElements * sizeof(double));
    }
    outFile.close();
}