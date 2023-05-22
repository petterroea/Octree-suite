#include "utils.h"

#include <ctime>
#include <sstream>
#include <fstream>
#include <iomanip>

std::string currentTimeAsString() {
    std::ostringstream oss;
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    oss << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    return oss.str();
}