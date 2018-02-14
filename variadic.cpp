#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include "plot.h"

int main()
{
    std::vector<double> x{1,2,3,4,5}, y{1,2,3,4,5};
    std::pair<std::string, std::string> sp {"marker","x"};
    std::pair<std::string, double> dp {"lw",2.0};
    Plt plt;
    plt.plot(x, y, sp, dp);

    /* std::vector<double> x{1,2,3,4,5}; */
    /* std::pair<std::string, std::string> sp {"marker","o"}; */
    /* std::pair<std::string, double> dp {"lw",2.0}; */
    /* Plt plt; */
    /* plt.plot(x, sp, dp); */
}
