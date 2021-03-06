//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <deque>

// pop_back() more than the number of elements in a deque

#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <cstdlib>
#include <deque>


int main() {
    std::deque<int> q;
    q.push_back(0);
    q.pop_back();
    q.pop_back();
    std::exit(1);
}
