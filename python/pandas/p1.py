import pandas as pd
import numpy as np

# The basic data structure is the series
s = pd.Series([1,2,3,4,5,6])
# print(dir(s)) literally made my terminal scroll. Series can do many things...

# Slicing works as expected
s0 = s[0] # type is np.int64
ss = s[2:4] # type is Series


# Underling the series is a np array
nps = s.values
assert type(nps) is np.ndarray





# Dataframe is a dict of series
df = pd.DataFrame({
        "x": pd.Series(np.arange(10)),
        "y": np.random.random(10),
})

# Regardless of what came in each col is now a series
assert type(df["y"]) is pd.core.series.Series

# You can also create one from a numpy structured array. **BUT**
# A numpy structured array is row major (data[i]['x'] is next to data[i]["y"])
# pandas dataframes are column major. Each col is contiguous.

# We can easily access a view of the underlying np array
npx = df["x"].values
assert type(npx) is np.ndarray
assert npx.flags["C_CONTIGUOUS"] and npx.flags["F_CONTIGUOUS"]

npx[0] = 10
assert df["x"][0] == 10 # Remember it is a view :)

# Because df is col major adding columns is fast and easy
# Contrast with numpy which makes adding keys to structured arrays hard
df["z"] = 1 # Creates a z col of the same length as the others and all vals = 1


# Of course there are downsides. Accessing a single row will be slower because the entries
# for each column are far away from each other. Imagine a 2d memory architecture. . .
print(df.iloc[5], "\n", df.loc[5])

assert (df.iloc[5] == df.loc[5]).all() # This works because the default index is 0, 1, ...

# Indexing data frames can be odd:
# Select column:        df[col]	        Series
# Select row by label:	df.loc[label]	Series
# Select row by pos:    df.iloc[loc]	Series
# Slice rows:           df[5:10]	DataFrame
# Boolean row:          df[bool_vec]	DataFrame


# So what's up with this index thing?

s = pd.Series(np.arange(100_000_000))

# It doesn't double memory usage! I guess if left as the default it is kept as a range or
# something something lazy evaluation something.
print(s.memory_usage() / 1e8)
print(s.memory_usage(index=False) / 1e8)

# You can use a col as the index
df.set_index("x", inplace=True)
