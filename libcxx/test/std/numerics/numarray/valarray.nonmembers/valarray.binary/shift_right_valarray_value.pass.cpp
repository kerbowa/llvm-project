//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// template<class T> valarray<T> operator>>(const valarray<T>& x, const T& y);

#include <valarray>
#include <cassert>
#include <cstddef>

int main()
{
    {
        typedef int T;
        T a1[] = { 8,  16, 24, 32, 40};
        T a2[] = { 1,   2,  3,  4,  5};
        const unsigned N = sizeof(a1)/sizeof(a1[0]);
        std::valarray<T> v1(a1, N);
        std::valarray<T> v2 = v1 >> 3;
        assert(v1.size() == v2.size());
        for (std::size_t i = 0; i < v2.size(); ++i)
            assert(v2[i] == a2[i]);
    }
}
