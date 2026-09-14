#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector> 
#include <string> 

typedef std::vector<std::vector<std::string>> CSVTable; 

template <typename T>
struct Result { 
    T value; 
    std::string errorMessage; 
    bool success; 

    static Result<T> Ok(T val) { return { val, "", true}; }
    static Result<T> Fail(std::string msg) { return {T(), msg, false}; }
}; 

namespace Utils {
    std::vector<std::string> split(const std::string& text); 
    CSVTable parse_csv(const std::string& filename);
    std::string makeTimestamp();
    size_t floorPowerOfTwo(size_t n);
    
};

#endif

