;;; prints a newline, followed by the object, followed by a space
(print "hello")
(print (list 1 2 3 4))

;;; ~D expects an integer, ~% gives a newline
(format t "~%")
(format t "1 + 2 = ~D~%" 3);
(format t "~A~%" "A is for strings");
