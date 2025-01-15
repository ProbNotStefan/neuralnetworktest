#include <iostream>
#include <vector>

int main() {
    std::vector<int> myVector;

    // Check the maximum possible size
    size_t max_size = myVector.max_size();
    std::cout << "Maximum size: " << max_size << std::endl;

    // Attempt to insert a very large number of elements (may exceed max_size)
    for (size_t i = 0; i < max_size * 2; ++i) { 
        myVector.push_back(i); 
    }

    return 0;
}