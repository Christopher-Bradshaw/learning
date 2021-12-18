#include <iostream>
#include <vector>

class Solution {
   public:
    int removeElement(std::vector<int>& nums, int val) {
        int moveTo = 0;
        for (int moveFrom = 0; moveFrom < nums.size(); ++moveFrom) {
            if (nums[moveFrom] != val) {
                nums[moveTo] = nums[moveFrom];
                moveTo++;
            }
        }
        return moveTo;
    }
};

int main() {
    Solution sol;

    std::vector<int> inp{1, 2, 2, 4};
    std::cout << sol.removeElement(inp, 2) << std::endl;
}