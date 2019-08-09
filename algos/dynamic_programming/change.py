"""
The classic dynamic programming problem - how to make change?
"""

memoize = {0: 0}

# Return the minimumm number of coins required to make this amount of change
def make_change(change, coins):

    try:
        return memoize[change]
    except KeyError:
        pass

    n_coins = []
    for coin in coins:
        if change - coin >= 0:
            n_coins.append(make_change(change - coin, coins) + 1)

    n_coins = min(n_coins)
    memoize[change] = n_coins
    return n_coins


def main():
    change = 25
    coins = [1, 5, 8]

    print(make_change(change, coins))


if __name__ == "__main__":
    main()
