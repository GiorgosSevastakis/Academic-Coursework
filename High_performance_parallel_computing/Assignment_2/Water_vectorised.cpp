#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cassert>
#include <math.h>
#include <chrono>

using namespace std;

const double deg2rad = acos(-1)/180.0; // pi/180 for changing degs to radians
double accumulated_forces_bond  = 0.;     // Checksum: accumulated size of forces
double accumulated_forces_angle = 0.;     // Checksum: accumulated size of forces
double accumulated_forces_non_bond = 0.;  // Checksum: accumulated size of forces

class Vec3 {
public:
    double x, y, z;
    // initialization of vector
    Vec3(double x, double y, double z): x(x), y(y), z(z) {}
    // size of vector
    double mag() const{
        return sqrt(x*x+y*y+z*z);
    }
    Vec3 operator-(const Vec3& other) const{
        return {x - other.x, y - other.y, z - other.z};
    }
    Vec3 operator+(const Vec3& other) const{
        return {x + other.x, y + other.y, z + other.z};
    }
    Vec3 operator*(double scalar) const{
        return {scalar*x, scalar*y, scalar*z};
    }
    Vec3 operator/(double scalar) const{
        return {x/scalar, y/scalar, z/scalar};
    }
    Vec3& operator+=(const Vec3& other){
        x += other.x; y += other.y; z += other.z;
        return *this;
    }
    Vec3& operator-=(const Vec3& other){
        x -= other.x; y -= other.y; z -= other.z;
        return *this;
    }
    Vec3& operator*=(double scalar){
        x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }
    Vec3& operator/=(double scalar){
        x /= scalar; y /= scalar; z /= scalar;
        return *this;
    }
};
Vec3 operator*(double scalar, const Vec3& y){
    return y*scalar;
}
Vec3 cross(const Vec3& a, const Vec3& b){
    return { a.y*b.z-a.z*b.y,
             a.z*b.x-a.x*b.z,
             a.x*b.y-a.y*b.x };
}
double dot(const Vec3& a, const Vec3& b){
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* atom class */
// class Atom {
// public:
//     // The mass of the atom in (U)
//     double mass;
//     double ep;            // epsilon for LJ potential
//     double sigma;         // Sigma, somehow the size of the atom
//     double charge;        // charge of the atom (partial charge)
//     std::string name;     // Name of the atom
//     // the position in (nm), velocity (nm/ps) and forces (k_BT/nm) of the atom
//     Vec3 p,v,f;
//     // constructor, takes parameters and allocates p, v and f properly
//     Atom(double mass, double ep, double sigma, double charge, std::string name) 
//     : mass{mass}, ep{ep}, sigma{sigma}, charge{charge}, name{name}, p{0,0,0}, v{0,0,0}, f{0,0,0}
//     {}
// };

/* a class for the bond between two atoms U = 0.5k(r12-L0)^2 */
class Bond {
public:
    double K;    // force constant
    double L0;   // relaxed length
    int a1, a2;  // the indexes of the atoms at either end
};

/* a class for the angle between three atoms  U=0.5K(phi123-phi0)^2 */
class Angle {
public:
    double K;
    double Phi0;
    int a1, a2, a3; // the indexes of the three atoms, with a2 being the centre atom
};

// /* molecule class */
// class Molecule {
// public:
//     std::vector<Atom> atoms;          // list of atoms in the molecule
//     std::vector<Bond> bonds;          // the bond potentials, eg for water the left and right bonds
//     std::vector<Angle> angles;        // the angle potentials, for water just the single one, but keep it a list for generality
// };

// ===============================================================================
// Two new classes arranging Atoms in a Structure-of-Array data structure
// ===============================================================================

/* atom class, represent N instances of identical atoms */
class Atoms {
public:
    // The mass of the atom in (U)
    double mass;
    double ep;            // epsilon for LJ potential
    double sigma;         // Sigma, somehow the size of the atom
    double charge;        // charge of the atom (partial charge)
    std::string name;     // Name of the atom
    // the position in (nm), velocity (nm/ps) and forces (k_BT/nm) of the atom
    std::vector<Vec3> p,v,f;
    // constructor, takes parameters and allocates p, v and f properly to have N_identical elements
    Atoms(double mass, double ep, double sigma, double charge, std::string name, size_t N_identical) 
    : mass{mass}, ep{ep}, sigma{sigma}, charge{charge}, name{name}, 
      p{N_identical, {0,0,0}}, v{N_identical, {0,0,0}}, f{N_identical, {0,0,0}}
    {}
};

/* molecule class for no_mol identical molecules */
class Molecules {
public:
    std::vector<Atoms> atoms;         // list of atoms in the N identical molecule
    std::vector<Bond> bonds;          // the bond potentials, eg for water the left and right bonds
    std::vector<Angle> angles;        // the angle potentials, for water just the single one, but keep it a list for generality
    int no_mol;
};

// ===============================================================================


/* system class */
class System {
public:
    Molecules molecules;          // all the molecules in the system /changed
    double time = 0;                          // current simulation time
};

class Sim_Configuration {
public:
    int steps = 10000;     // number of steps
    int no_mol = 256;        // number of molecules
    double dt = 0.0005;    // integrator time step
    int data_period = 100; // how often to save coordinate to trajectory
    std::string filename = "trajectory.txt";   // name of the output file with trajectory
    // system box size. for this code these values are only used for vmd, but in general md codes, period boundary conditions exist

    // simulation configurations: number of step, number of the molecules in the system, 
    // IO frequency, time step and file name
    Sim_Configuration(std::vector <std::string> argument){
        for (long unsigned int i = 1; i<argument.size() ; i += 2){
            std::string arg = argument.at(i);
            if(arg=="-h"){ // Write help
                std::cout << "MD -steps <number of steps> -no_mol <number of molecules>"
                          << " -fwrite <io frequency> -dt <size of timestep> -ofile <filename> \n";
                exit(0);
                break;
            } else if(arg=="-steps"){
                steps = std::stoi(argument[i+1]);
            } else if(arg=="-no_mol"){
                no_mol = std::stoi(argument[i+1]);
            } else if(arg=="-fwrite"){
                data_period = std::stoi(argument[i+1]);
            } else if(arg=="-dt"){
                dt = std::stof(argument[i+1]);
            } else if(arg=="-ofile"){
                filename = argument[i+1];
            } else{
                std::cout << "---> error: the argument type is not recognized \n";
            }
        }

        dt /= 1.57350; /// convert to ps based on having energy in k_BT, and length in nm
    }
};

// Given a bond, updates the force on all atoms correspondingly
void UpdateBondForces(System& sys){
    // for (Molecules& molecule : sys.molecules) //changed
    // Loops over the (2 for water) bond constraints

    for (Bond& bond : sys.molecules.bonds){
        auto& atom1=sys.molecules.atoms[bond.a1];
        auto& atom2=sys.molecules.atoms[bond.a2];
        for (int i = 0 ; i<sys.molecules.no_mol ; i++){
            Vec3 dp  = atom1.p[i]-atom2.p[i];
            Vec3 f   = -bond.K*(1-bond.L0/dp.mag())*dp;
            atom1.f[i] += f;
            atom2.f[i] -= f; 
            accumulated_forces_bond += f.mag();
        }
    }
}

// Iterates over all bonds in molecules (for water only 2: the left and right)
// And updates forces on atoms correpondingly
void UpdateAngleForces(System& sys){
    for (Angle& angle : sys.molecules.angles){ ////////////////////////////////////////////////// CHANGEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        //====  angle forces  (H--O---H bonds) U_angle = 0.5*k_a(phi-phi_0)^2
        //f_H1 =  K(phi-ph0)/|H1O|*Ta
        // f_H2 =  K(phi-ph0)/|H2O|*Tc
        // f_O = -f1 - f2
        // Ta = norm(H1O x (H1O x H2O))
        // Tc = norm(H2O x (H2O x H1O))
        //=============================================================
        auto& atom1=sys.molecules.atoms[angle.a1];
        auto& atom2=sys.molecules.atoms[angle.a2];
        auto& atom3=sys.molecules.atoms[angle.a3];
        for (int i = 0 ; i<sys.molecules.no_mol ; i++)
        {
            Vec3 d21 = atom2.p[i]-atom1.p[i];     
            Vec3 d23 = atom2.p[i]-atom3.p[i];    

            // phi = d21 dot d23 / |d21| |d23|
            double norm_d21 = d21.mag();
            double norm_d23 = d23.mag();
            double phi = acos(dot(d21, d23) / (norm_d21*norm_d23));

            // d21 cross (d21 cross d23)
            Vec3 c21_23 = cross(d21, d23);
            Vec3 Ta = cross(d21, c21_23);
            Ta /= Ta.mag();

            // d23 cross (d23 cross d21) = - d23 cross (d21 cross d23) = c21_23 cross d23
            Vec3 Tc = cross(c21_23, d23);
            Tc /= Tc.mag();

            Vec3 f1 = Ta*(angle.K*(phi-angle.Phi0)/norm_d21);
            Vec3 f3 = Tc*(angle.K*(phi-angle.Phi0)/norm_d23);

            atom1.f[i] += f1;
            atom2.f[i] -= f1+f3;
            atom3.f[i] += f3;

            accumulated_forces_angle += f1.mag() + f3.mag();
        }
    }
}

// Iterates over all atoms in both molecules
// And updates forces on atoms correpondingly
// void UpdateNonBondedForces(System& sys){
//     /* nonbonded forces: only a force between atoms in different molecules
//        The total non-bonded forces come from Lennard Jones (LJ) and coulomb interactions
//        U = ep[(sigma/r)^12-(sigma/r)^6] + C*q1*q2/r */
//     double KC = 80*0.7;
//     for (long unsigned int i = 0;   i < sys.molecules.no_mol; i++)
//     for (long unsigned int j = i+1; j < sys.molecules.no_mol; j++)
//     for (auto& atom1 : sys.molecules.atoms)
//     for (auto& atom2 : sys.molecules.atoms){ // iterate over all pairs of atoms, similar as well as dissimilar
//             Vec3 dp = atom1.p[i]-atom2.p[j];

//             double r  = dp.mag();                   
//             double r2 = r*r;
//             double ep = sqrt(atom1.ep*atom2.ep); // ep = sqrt(ep1*ep2)
//             double sigma = 0.5*(atom1.sigma+atom2.sigma);  // sigma = (sigma1+sigma2)/2
//             double q1 = atom1.charge;
//             double q2 = atom2.charge;

//             double sir = sigma*sigma/r2; // crossection**2 times inverse squared distance
//                       // Coulomb prefactor
//             Vec3 f = ep*(12*pow(sir,6)-6*pow(sir,3))*sir*dp + KC*q1*q2/(r*r2)*dp; // LJ + Coulomb forces
//             atom1.f[i] += f;
//             atom2.f[j] -= f;

//             accumulated_forces_non_bond += f.mag();
//         }
// }
void UpdateNonBondedForces(System& sys){
    /* nonbonded forces: only a force between atoms in different molecules
       The total non-bonded forces come from Lennard Jones (LJ) and coulomb interactions
       U = ep[(sigma/r)^12-(sigma/r)^6] + C*q1*q2/r */
    
    auto& molecules = sys.molecules;
    for (auto& atoms1 : molecules.atoms) {
        for (auto& atoms2 : molecules.atoms) {
            for (int i = 0; i < molecules.no_mol; i++) {
                #pragma omp simd
                for (int j = i + 1; j < molecules.no_mol; j++) {
                    Vec3 dp = atoms1.p[i]-atoms2.p[j];
                    double r  = dp.mag();                   
                    double r2 = r*r;
                    double ep = sqrt(atoms1.ep*atoms2.ep); // ep = sqrt(ep1*ep2)
                    double sigma = 0.5*(atoms1.sigma+atoms2.sigma);  // sigma = (sigma1+sigma2)/2
                    double q1 = atoms1.charge;
                    double q2 = atoms2.charge;
                    double sir = sigma*sigma/r2; // crossection**2 times inverse squared distance
                    double KC = 80*0.7;          // Coulomb prefactor
                    Vec3 f = ep*(12*pow(sir,6)-6*pow(sir,3))*sir*dp + KC*q1*q2/(r*r2)*dp; // LJ + Coulomb forces
                    atoms1.f[i] += f;
                    atoms2.f[j] -= f;
                    accumulated_forces_non_bond += f.mag();
                }
            }
        }
    } 
}

// integrating the system for one time step using Leapfrog symplectic integration
void Evolve(System &sys, Sim_Configuration &sc){

    // Kick velocities and zero forces for next update
    // Drift positions: Loop over molecules and atoms inside the molecules
    // for (Molecules& molecule : sys.molecules)
    for (auto& atom : sys.molecules.atoms){
        for (int i = 0 ; i<sys.molecules.no_mol ; i++) {
        
        
        atom.v[i] += sc.dt/atom.mass*atom.f[i];    // Update the velocities
        atom.f[i]  = {0,0,0};                   // set the forces zero to prepare for next potential calculation
        atom.p[i] += sc.dt* atom.v[i];             // update position
        }
    }

    // Update the forces on each particle based on the particles positions
    // Calculate the intermolecular forces in all molecules
    UpdateBondForces(sys);
    UpdateAngleForces(sys);
    // Calculate the intramolecular LJ and Coulomb potential forces between all molecules
    UpdateNonBondedForces(sys);

    sys.time += sc.dt; // update time
}

// Setup one water molecule
System MakeWater(int N_molecules){
    //===========================================================
    // creating water molecules at position X0,Y0,Z0. 3 atoms
    //                        H---O---H
    // The angle is 104.45 degrees and bond length is 0.09584 nm
    //===========================================================
    // mass units of dalton
    // initial velocity and force is set to zero for all the atoms by the constructor
    const double L0 = 0.09584;
    const double angle = 104.45*deg2rad;    

    //         mass    ep    sigma charge name
    // Atom Oatom(16, 0.65,    0.31, -0.82, "O");  // Oxygen atom
    // Atom Hatom1( 1, 0.18828, 0.238, 0.41, "H"); // Hydrogen atom
    // Atom Hatom2( 1, 0.18828, 0.238, 0.41, "H"); // Hydrogen atom

    Atoms Oatoms(16, 0.65,    0.31, -0.82, "O",N_molecules);
    Atoms Hatoms1(1, 0.18828, 0.238, 0.41, "H",N_molecules);
    Atoms Hatoms2(1, 0.18828, 0.238, 0.41, "H",N_molecules);

    // bonds beetween first H-O and second H-O respectively
    std::vector<Bond> waterbonds = {
        { .K = 20000, .L0 = L0, .a1 = 0, .a2 = 1},
        { .K = 20000, .L0 = L0, .a1 = 0, .a2 = 2}
    };

    // angle between H-O-H
    std::vector<Angle> waterangle = {
        { .K = 1000, .Phi0 = angle, .a1 = 1, .a2 = 0, .a3 = 2 }
    };   

    System sys;
    for (int i = 0; i < N_molecules; i++){
        Vec3 P0{i * 0.2, i * 0.2, 0};
        Oatoms.p[i]  = {P0.x, P0.y, P0.z};
        Hatoms1.p[i] = {P0.x+L0*sin(angle/2), P0.y+L0*cos(angle/2), P0.z};
        Hatoms2.p[i] = {P0.x-L0*sin(angle/2), P0.y+L0*cos(angle/2), P0.z};
        // Oatoms.p.push_back({P0.x, P0.y, P0.z});
        // Hatoms1.p.push_back({P0.x+L0*sin(angle/2), P0.y+L0*cos(angle/2), P0.z});
        // Hatoms2.p.push_back({P0.x-L0*sin(angle/2), P0.y+L0*cos(angle/2), P0.z});
        // std::vector<Atom> atoms {Oatoms, Hatoms1, Hatoms2};

    }
    // sys.molecules({atoms, waterbonds, waterangle});
    sys.molecules.atoms.push_back(Oatoms);
    sys.molecules.atoms.push_back(Hatoms1);
    sys.molecules.atoms.push_back(Hatoms2);

    sys.molecules.bonds= waterbonds;
    sys.molecules.angles= waterangle;
    sys.molecules.no_mol=N_molecules;

    // Store atoms, bonds and angles in Water class and return
    return sys;
}

// Write the system configurations in the trajectory file.
void WriteOutput(System& sys, std::ofstream& file){  
    // Loop over all atoms in model one molecule at a time and write out position
    for (auto& atom : sys.molecules.atoms){
        for(int i = 0 ; i < sys.molecules.no_mol ; i++){
            file << sys.time << " " << atom.name << " " 
                << atom.p[i].x << " " 
                << atom.p[i].y << " " 
                << atom.p[i].z << '\n';
        }
    }
}


//======================================================================================================
//======================== Main function ===============================================================
//======================================================================================================
int main(int argc, char* argv[]){    
    Sim_Configuration sc({argv, argv+argc}); // Load the system configuration from command line data
    
    System sys  = MakeWater(sc.no_mol);   // this will create a system containing sc.no_mol water molecules
    std::ofstream file(sc.filename); // open file

    WriteOutput(sys, file);    // writing the initial configuration in the trajectory file
    
    auto tstart = std::chrono::high_resolution_clock::now(); // start time (nano-seconds)
    
    // Molecular dynamics simulation
    for (int step = 0;step<sc.steps ; step++){

        Evolve(sys, sc); // evolving the system by one step
        if (step % sc.data_period == 0){
            //writing the configuration in the trajectory file
            WriteOutput(sys, file);
        }
    }

    auto tend = std::chrono::high_resolution_clock::now(); // end time (nano-seconds)

    std::cout <<  "Elapsed time:" << std::setw(9) << std::setprecision(4)
              << (tend - tstart).count()*1e-9 << "\n";
    std::cout <<  "Accumulated forces Bonds   : "  << std::setw(9) << std::setprecision(5) 
              << accumulated_forces_bond << "\n";
    std::cout <<  "Accumulated forces Angles  : "  << std::setw(9) << std::setprecision(5)
              << accumulated_forces_angle << "\n";
    std::cout <<  "Accumulated forces Non-bond: "  << std::setw(9) << std::setprecision(5)
              << accumulated_forces_non_bond << "\n";
}
