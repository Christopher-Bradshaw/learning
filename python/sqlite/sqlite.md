# Sqlite

While we are on the python interface it is probably worth also covering the shell. This is launched with `sqlite3`.

**Remeber that you need to terminate queries with a `;`**

## Dot commands and sqlite_master

These are things the command line program understands. These won't work as e.g. a query from the python implementation.

* `.open FILENAME` - open the DB in FILENAME
* `.tables` - list all tables
* `.schema` - list all tables with their schema
* `.indexes` - list all indexes

These tend to correspond to queries on the readonly `sqlite_master` table. This table has schema:

* `type text` - what the type of this entry is, e.g. table or index
* `name text` - the name of this table or index
* `table_name text` - which table this index is on (or the same as `name` if type is table)
* `rootpage int` - not sure...,
* `sql text` - the `CREATE TABLE` or `CREATE INDEX` command

So for example `.tables` is equivalent to `select name from sqlite_master where type="table"`

## Basic SQL stuff

Let's say we want to create a new table
```
CREATE TABLE tablename (col1 int, col2 string, ...)
```

Now, we can insert into that table,
```
INSERT into tablename values (2, "hehe", ...)
```

To get info about the columns in a table,
```
pragma table_info(tablename)
```

And to read out the data in a table
```
SELECT col1,col2 FROM tablename LIMIT 2;
```

## Indexes

Making things fast often comes down to having the right indexes.

`CREATE INDEX indexname ON tablename(col1, col2, ...)`

But how do we know what the right indexes are? Pick the query you want to optimize and see how the DB will run it.

`EXPLAIN QUERY PLAN SELECT * from tablename where col=val`

Some examples of what you will get:
* `SCAN TABLE points` - no index, O(n) scan
* `SEARCH TABLE points USING INDEX x (x=?)` - use the index on x
* `SEARCH TABLE points USING COVERING INDEX x (x=?)` - search using the index on x and the columns we requested are available in the index so we don't even need to access the row.
