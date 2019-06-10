# An alternative to pdb, pudb seems pretty cool!

# We can get into pudb using `breakpoint()` if we define
# PYTHONBREAKPOINT=pudb.set_trace python3.7 debugger.py

def main():
    import pudb; pudb.set_trace()
    a = 1
    b = 2
    c = a + b
    print("hi")
    return c

if __name__ == "__main__":
    main()
