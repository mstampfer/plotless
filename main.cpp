#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include "plot.h"
#include <numeric>
#include <Eigen/Dense>
using namespace Eigen;
auto range(double start, double end)
{
    std::vector<double> v(end-start);
    std::iota(v.begin(),v.end(), start);
    return std::move(v);
}
int main()
{
    Plt plt;
    typedef Matrix<double, 4, 1> ColVector4d;
    
    ColVector4d x(range(1,5).data()); 
    std::vector<ColVector4d> v{x, x*1.5, x, x*3.0, x, x/3.0};
    Args<ColVector4d> args(v);
    
    auto sp = std::make_pair("marker","x");
    auto dp = std::make_pair("lw",2.0);
    Kwargs kwargs({sp, dp}); 

    plt.plot(args, kwargs);
    plt.axis({0.0,5.0,-1.0,13.0});
    plt.xlabel("This is the X axis");
    plt.ylabel("This is the Y axis");
    plt.grid(true);
    plt.show();

    return 0;
}
