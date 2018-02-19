#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include "plot.h"
#include <numeric>

auto range(double start, double end)
{
    std::vector<double> v(end-start);
    std::iota(v.begin(),v.end(), start);
    return v;
}
int main()
{
    Plt plt;
    RowVector3d x(range(1,5).data()); 
    std::initializer_list<RowVector3d> il{x, x*1.5, x, x*3.0, x, x/3.0};
    Args<RowVector3d> args(il);
    
    auto sp = std::make_pair("marker","x");
    auto dp = std::make_pair("lw",2.0);
    Kwargs<char, double> kwargs(sp, dp);
    plt.plot(args, kwargs);
    
    plt.axis({0.0,5.0,-1.0,13.0});
    plt.show();

    return 0;
}
