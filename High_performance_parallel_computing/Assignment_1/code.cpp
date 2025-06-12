#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;

int main(){

    // Declaration and Initialization of the problem's parameters.
    const int N = 1000;
    const double b = 1/5., g = 1/10.;

    // Declaration and Initialization of the variables S, I and R and Declaration of the rates and individual update variables.
    double S = N-1 , S_dot, S_new;
    double I = 1, I_dot, I_new;
    double R = 0, R_dot, R_new;

    // Declaration and Initialization of time parameters.
    const double T = 300, dt = 0.1;
    int Nt = 0;
    double t_i = 0;

    // Declaration of the vectors that will store the values.
    vector<double> t;
    vector<double> S_store;
    vector<double> I_store;
    vector<double> R_store;

    // Initial conditions in the vectors
    t.push_back(0);
    S_store.push_back(S);
    I_store.push_back(1);
    R_store.push_back(0);

    // Calculating total number of steps.
    Nt = floor(T / dt);
    
    // For-loop which implements the forward Euler method to compute the S, I and R variables over time.
    for(int i = 0; i < Nt; i++){ 
        // Counting the time.
        t_i += dt;
        t.push_back(t_i);

        // Updating and storing S. 
        S_dot = -b/N * I * S;
        S_new = S + S_dot * dt;
        S_store.push_back(S_new);

        // Updating aand storing I. 
        I_dot = b/N * I * S - g * I;
        I_new = I + I_dot * dt;
        I_store.push_back(I_new);

        // Updating and storing R.
        R_dot = g * I;
        R_new = R + R_dot * dt;
        R_store.push_back(R_new);

        // Updating the variables for the next iteration.
        S = S_new;
        I = I_new;
        R = R_new;

}   

// We import the variables S, I and R, as well as the time t in a file, namely "sir_output.txt", with headers.  
ofstream myfile;
myfile.open("sir_output.txt");
myfile << "Time(days)" << " " << "Susceptible" << " " << "Infected" << " " << "Recovered" << endl;
for (int j = 0; j <= Nt; j++) {
    myfile << t[j] << " " << S_store[j] << " " << I_store[j] << " " << R_store[j] << endl;
    }

myfile.close();

return 0;
}