import pandas as pd
import numpy as np

# Can think of a dataframe as a dict of series
df = pd.DataFrame({
        "x": pd.Series(np.arange(10)),
        "y": np.random.random(10),
})

# Regardless of what came in each col is now a series
# Also note how we get column names
for col in list(df):
    assert type(df[col]) is pd.core.series.Series

# You can also create a dataframe from a numpy structured array. **BUT**
# A numpy structured array is row major (data[i]['x'] is next to data[i]["y"])
# pandas dataframes are column major. Each col is contiguous.

# We can easily access a view of the underlying np array
npx = df["x"].values
assert type(npx) is np.ndarray
assert npx.flags["C_CONTIGUOUS"] and npx.flags["F_CONTIGUOUS"]

npx[0] = 10
assert df["x"][0] == 10 # Remember it is a view!

# Because df is col major adding columns is fast and easy
# Contrast with numpy which makes adding keys to structured arrays hard
df["z"] = 1 # Creates a z col of the same length as the others and all vals = 1


print("This is our dataframe")
print(df)
print()

# Of course there are downsides. Accessing a single row will be slower because the entries
# for each column are far away from each other. Imagine a 2d memory architecture. . .
print("We can access rows using either the the location or the label")
print("In this case they are the same (by default the label is just the location")

print("by location")
print(df.iloc[5])
print("by label")
print(df.loc[5])

assert (df.iloc[5] == df.loc[5]).all() # This works because the default index is 0, 1, ...

# Indexing data frames can be odd:
# Select column:        df[col]	        Series
# Select row by label:	df.loc[label]	Series
# Select row by pos:    df.iloc[loc]	Series
# Slice rows:           df[5:10]	DataFrame
# Boolean row:          df[bool_vec]	DataFrame


# So what's up with this index thing?

length = int(1e8)
s = pd.Series(np.arange(length))

# It doesn't double memory usage! I guess if left as the default it is kept as a range or
# something something lazy evaluation something.
print(s.memory_usage() / length)
print(s.memory_usage(index=False) / length)

# You can use a col as the index
df.set_index("y", inplace=True)

# Note how y is not longer a column!
print(list(df))

# We can select it with df.index
y5 = df.index.values[5]
print(df.loc[y5])
