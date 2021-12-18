#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

class Solution {
   public:
    bool isValid(std::string s) {
        // Remember chars need single quotes (strings use double)
        std::set<char> opening{'(', '{', '['};
        std::map<char, char> closeToOpen{{')', '('}, {'}', '{'}, {']', '['}};
        std::vector<char> stack;

        for (char c : s) {
            if (opening.find(c) != opening.end()) {
                stack.push_back(c);
            } else {
                if (stack.size() == 0 || closeToOpen[c] != stack.back()) {
                    return false;
                }
                stack.pop_back();
            }
        }
        return (stack.size() == 0);
    };
};

int main(int argc, char **argv) {
    Solution sol;
    std::cout << sol.isValid("()") << std::endl;
    std::cout << sol.isValid("(") << std::endl;
    std::cout << sol.isValid("]") << std::endl;
    std::cout << sol.isValid("())(") << std::endl;
    std::cout << sol.isValid("([)]") << std::endl;
    std::cout << sol.isValid("([])") << std::endl;
    return 0;
};