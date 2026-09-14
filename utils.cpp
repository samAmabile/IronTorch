#include "utils.hpp"
#include <fstream>
#include <unordered_set> 
#include <ctime>
#include <chrono>
#include <iomanip>




namespace Utils {
    std::vector<std::string> split(const std::string& text){
        std::unordered_set<char> delimiters = {' ', ';', '{', '}', '[', ']', '(', ')', ':', '%', '=', '+', '#', '^', '/', '&', '|', '@', '!', '-', '<', '>'};
        std::string buffer = ""; 
        std::vector<std::string> tokens; 
        for (char c: text){
            if (delimiters.find(c) != delimiters.end()){
                if (!buffer.empty()) tokens.push_back(buffer); 
                tokens.push_back(std::string(1, c));
                buffer = ""; 
            } else {
                buffer += c; 
            }
        }

        return tokens; 
    }

    CSVTable parse_csv(const std::string& filename){
        std::ifstream csv(filename); 
        CSVTable table; 
        std::vector<std::string> row; 
        std::string cell; 
        bool inQuotes = false; 
        char c; 

        while (csv.get(c)) {
            if (inQuotes) {
                if (c == '"') {
                    if (csv.peek() == '"') {
                        cell += c; 
                        csv.get(); 
                    }else {
                        inQuotes = false; 
                    }
                }else {
                    cell += c; 
                }
            }else {
                if (c == '"') {
                    inQuotes = true; 
                }else if (c == ',') {
                    row.push_back(cell); 
                    cell = ""; 
                }else if (c == '\n' || c == '\r') {
                    row.push_back(cell); 
                    table.push_back(row); 
                    row.clear(); 
                    cell = "";
                    if (c == '\r' && csv.peek() == '\n') csv.get(); 
                }else {
                    cell += c; 
                }
            }
        }

        if (!cell.empty() || !row.empty()) {
            row.push_back(cell); 
            table.push_back(row); 
        }

        return table; 

    }

    std::string makeTimestamp() {
        auto now = std::chrono::system_clock::now(); 

        std::time_t timeT = std::chrono::system_clock::to_time_t(now);

        std::tm localTime; 
#if defined(_WIN32)
        localtime_s(&localTime, &timeT); 
#else
        localtime_r(&timeT, &localTime); 
#endif 

        std::stringstream ss; 
        ss << std::put_time(&localTime, "%Y-%m-%d_%H-%M-%S");

        return ss.str(); 
    }

    size_t floorPowerOfTwo(size_t n) {
        if (n == 0) return 0; 

        size_t p = 1; 
        while ((p << 1) <= n) {
            p <<= 1; 
        }

        return p; 
    }



}
