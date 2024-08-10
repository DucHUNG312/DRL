#include "lab/utils/file.h"

namespace lab
{
namespace utils
{
std::string join_paths(std::string head, const std::string& tail)
{
    if (head.back() != '/') {
        head.push_back('/');
    }
    head += tail;
    return head;
}
}
}