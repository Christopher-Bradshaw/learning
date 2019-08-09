"""
The classic dynamic programming problem - how to make change?
"""

# Store the base case - to make 0 requires 0 coins.
memoize = {0: (0, [])}


def make_change(change, coins):

    # Check if we have already solved this problem (or are at the base case).
    # This is needed to get O(n) scaling.
    try:
        return memoize[change]
    except KeyError:
        pass

    # Break the problem into all its possible sub-problems and solve those.
    n_coins = []
    for coin in coins:
        if change - coin >= 0:
            sub_problem_n, sub_problem_which = make_change(change - coin, coins)
            n_coins.append(((sub_problem_n + 1), (sub_problem_which + [coin])))

    # Choose which of the solutions is best
    min_idx = None
    for i, _ in enumerate(n_coins):
        if min_idx is None:
            min_idx = i
        elif n_coins[i][0] < n_coins[min_idx][0]:
            min_idx = i

    # Memoize the solution to this problem (as it might potentially be a subproblem later) and return it
    memoize[change] = n_coins[min_idx]
    return memoize[change]


def main():
    change = 93
    coins = [1, 5, 8, 27, 43]

    print(f"{change}: {make_change(change, coins)}")


if __name__ == "__main__":
    main()
