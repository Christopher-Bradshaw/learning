import sqlite3

import numpy as np

conn = sqlite3.connect("test.db")
c = conn.cursor()

# Cleanup from previous runs
c.execute("DROP TABLE IF EXISTS flights")
c.execute("DROP TABLE IF EXISTS points")

# Create and insert
c.execute("CREATE TABLE flights (depart text, arrive text, length real)")

c.execute("INSERT into flights VALUES ('LHR', 'SFO', 10.5)")
c.execute("INSERT into flights VALUES ('JFK', 'ORT', 14.5)")

# These inserts are immediately visible to this connection
c.execute("SELECT * from flights")
selected = c.fetchall()
assert len(selected) == 2

# But not to a new connection!
c2 = sqlite3.connect("test.db").cursor()
c2.execute("SELECT * from flights")
new_conn_selected = c2.fetchall()
assert len(new_conn_selected) == 0

# Commit them
conn.commit()

# And now they are here
c2.execute("SELECT * from flights")
new_conn_selected = c2.fetchall()
assert new_conn_selected == selected

# Note the format - each row is a tuple.
# And if more than one was selected we get a list of them.
print(selected)


# What about inserting from e.g. a numpy array
np_points = np.arange(10).view([("x", int), ("y", int)])
c.execute("CREATE TABLE points (x int, y int)")

# Put a single entry (tuple) in
# Note how we don't use python string formatting to create the string!
# Doing it with the ? will (I think) properly escape everything
c.execute("INSERT into points VALUES (?, ?)", (7, 3))
# Put multiple in - note that this needs a list of tuples
c.executemany("INSERT into points VALUES (?, ?)", np_points.tolist())

# What if want to be able to efficiently query on x?
c.execute("CREATE INDEX x ON points(x)")

conn.commit()
conn.close()
