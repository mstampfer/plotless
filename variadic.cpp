#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include "plot.h"

int main()
{
    Plt plt;
    std::vector<double> v1{1.,2.,3.};
    std::vector<double> v2{1.,2.,3.};
    std::vector<double> v3{1.,2.,3.};
    std::initializer_list<std::vector<double>> il{v1};
    /* std::initializer_list<std::vector<double>> il{v1, v2, v3}; */
    Args<std::vector<double>> args(il);
    
    std::pair<std::string, std::string> sp {"marker","x"};
    std::pair<std::string, double> dp {"lw",2.0};
    Kwargs<char, int> kwargs(sp, dp);
    plt.plot(args.args, kwargs.kwargs);

    /* std::vector<double> x{1,2,3,4,5}, y{1,2,3,4,5}; */
    /* std::pair<std::string, std::string> sp {"marker","x"}; */
    /* std::pair<std::string, double> dp {"lw",2.0}; */
    /* Plt plt; */
    /* plt.plot(x, y, sp, dp); */

    /* std::vector<double> x{1,2,3,4,5}; */
    /* std::pair<std::string, std::string> sp {"marker","o"}; */
    /* std::pair<std::string, double> dp {"lw",2.0}; */
    /* Plt plt; */
    /* plt.plot(x, sp, dp); */
    return 0;
}
