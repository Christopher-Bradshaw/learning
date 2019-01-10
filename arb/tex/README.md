# Tex


## Simple TeX docs

From Chapter 6 of the TeXbook

### The simplest possible TeX

`tex simplest && dvipdfm simplest` and see the output in `simplest.pdf`.

Wow, so easy!


### A slightly more complicated example

See `story.tex`. Nothing really surprising here but shows some basic formatting things.

### Overful \hbox

Edit `story.tex` by adding `\hsize=2in` at the start. This limits the horizontal width to 2 inches.

On compilation we not get errors complaining of an `overfull \hbox ... too wide`. In the compiled doc we also see black boxes showing where this happened. What's up?

TeX justifies the text, but has limits on the minimum and maximum width of spaces. There is no way for it to satisfy these criteria (fixed line length, valid space size) for these short lines.

Let's look at the error in detail. While a summary is printed on the command line the full error is in `story.log`. All the lines are too wide, and the log shows the places that TeX considered hyphenation. The problem here is that TeX's default settings don't work for really short lines - there just isn't a way to hyphenate these lines to have good spacing at this width. We need to allow it to be more flexible (allow larger/smaller spaces).

TeX rates the spacing of each line by assigning it a `badness` score. The ideal spacing comes with a score of 0, and as spacing gets tight/too lose scores increase. By default TeX requires a badness of less than 200. We can change this by setting `tolerance=X`. Another option ([Tufte approved](https://twitter.com/edwardtufte/status/732696112818184193)) is to just set `\raggedright` which removes the justification requirement and makes everything much easier for TeX. If we prefer some overful rather than increased spacing `\hfuzz=1pt` ignores all errors up to the give size (by default this is 0.1pt).


### Handling errors

The previous example showed how to handle warnings about poor performance. What about actual breaking errors? See `sorry.tex`.

```
! Undefined control sequence.   # ! means an error, the text exaplins what error it was
l.3 \vship                      # The error was on line 3. The content on this line is what TeX has read so far
           1in                  # and the content on this line is what is still left on the line
?                               # TeX requests advice on how to proceed
```

If you are like me before I read this you now frantically hit `C-c C-d` until TeX dies and then you go look at that line. However, there are better options.

* `?` - Ask TeX for a list of options.
* `<Enter>` - Tell TeX to proceed by doing what it thinks is best.
* `S` - As above but don't stop for future errors either.
* `H` - Print the informal message that explains what went wrong.
* `X` - Exit, after finishing up the log file and any completed previous pages.
* `E` - Open the file in an editor with the cursor at the right place.

I've left out some options that either really agressively ignore future errors or allow you to edit the file from within TeX. We should see errors and rather than learning how to edit in TeX, just drop into the editor and recompile.

Let's assume we fixed this first error and reran TeX.

```
! Undefined control sequence.
<argument> \bf A SHORT \ERROR           # What TeX has read so far - so the error is somewhere here
                              STORY     # What is to come
\centerline #1->\line {\hss #1          # What TeX was doing - running centerline. The stuff after -> is just the expansions of centerline! TeX had got to the argument when something went wrong.
                              \hss }    # And after the argument it was going to do this.
l.4 \centerline{\bf A SHORT \ERROR STORY}
                                        # Note the empty line here because TeX read everything on this line.
```

This looks crazy but is actually just a stack trace! Reading from bottom to top you get the line the error was on, then the expansion of the function centerline, then the expansion of the argument to that, which is where the actual error happened!

The best clues to what went wrong (according to the TeXbook) are:

Usually on the bottom line (since that is what you typed) and on the top line (since that is what triggered the error message).

## Reserved Characters

