(print "Vectors")
;;; Vectors. Think of them as arrays
(let ((x (vector 1 2)))
  (print x)
  ;; Index into the array to read and modify
  (print (aref x 1))
  (setf (aref x 1) 3)
  (print x)
)

;;; Create an new vector
(let ((x (make-array 5 :initial-element 1)))
  (print x)
)


(print "Lists")
(let ((x (list 1 2)))
  (print x)
  (print (nth 0 x))
)

(print "PLists") ; Property list
(let ((x (list :name "bob" :age 33)))
  (print x)
  (print (getf x :name))
)

(let* (
  (pine "pin")
  (numbers "numbers")
  (x (list :pine "cones" :numbers (list 1 2 3)))
)
  (print x)
  (print (getf x :pine))
)
