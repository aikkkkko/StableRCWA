#include <cmath>
#include <iostream>
#include "stablercwa/rcwa1d_conical.hpp"
using namespace stablercwa;
namespace dc=rcwa::detail::conical;
int main() {
    int failed=0;
    auto check=[&](bool ok) { if (!ok) ++failed; };
    auto fails=[&](auto f) { try { f(); return false; } catch (rcwa::detail::NumericalFailure&) { return true; } };
    check(dc::triangular_exp(Matrix(0,0)).rows()==0);
    check(dc::triangular_root(Matrix(0,0)).rows()==0);
    Matrix t(2,2); t(0,0)=t(1,1)=4.; t(0,1)=3.;
    const Matrix g=dc::triangular_root(t);
    check(std::abs(g(0,1)-.75)<1e-15);
    const Matrix e=dc::triangular_exp(g*cplx(-.3));
    check(std::abs(e(0,1)+.225*std::exp(-.6))<1e-14);
    Matrix dynamic(2,2); dynamic(0,0)=-1e20; dynamic(1,1)=-1.;
    const Matrix de=dc::triangular_exp(dynamic);
    check(de(0,0)==cplx{} && std::abs(de(1,1)-std::exp(-1.))<1e-15);
    Matrix one(1,1); one(0,0)=1e-200;
    check(std::abs(dc::triangular_root(one)(0,0)-1e-100)<1e-114);
    one(0,0)=1e200;
    check(std::abs(dc::triangular_root(one)(0,0)/1e100-1.)<1e-14);
    one(0,0)=0.; check(fails([&]{ dc::triangular_root(one); }));
    one(0,0)=1000.; check(fails([&]{ dc::triangular_exp(one); }));
    one(0,0)=std::numeric_limits<double>::infinity();
    check(fails([&]{ dc::triangular_root(one); }));
    check(fails([&]{ dc::triangular_exp(one); }));
    std::cout << "matrix function failures: " << failed << '\n';
    return failed ? 1 : 0;
}
