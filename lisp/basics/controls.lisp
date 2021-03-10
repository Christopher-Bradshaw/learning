;;; The basic if-else
(print (if T "True" "False"))
(print (if nil "True" "False"))
(print (if (> 3 2) "True" "False"))

;;; Cond, the if-elif-elif-...-else construct
(let ((x 5))
  (cond
    ((eql x 1) (print "x was 1"))
    ((eql x 2) (print "x was 2"))
    ((eql x 3) (print "x was 3"))
    (T (print "x was large"))
  )
)
