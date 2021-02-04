(print "Declarations")

;;; Declare block scoped variables
(let ((x 10) (y 11))
  (print x)
  (print y)
)

;;; Declare block scoped variables where later vars can refer
;;; back to earlier ones
(let* ((x 10) (y (+ x 1)))
  (print x)
  (print y)
)

;;; Note we could achieve the same as let* with nested lets
(let ((x 10))
  (let ((y (+ x 1)))
    (print x)
    (print y)
  )
)

(print "Modifications")
;;; We can modify variables
(let ((x 10) (y 11))
  (setf x 12)
  (print x)
  (setf x 13 y 12)
  (print x)
  (print y)
)

(print "Globals")
;;; Declare a global (aka dynamic/special) variable
;;; By convention these are wrapped in *s
;;; This seems to be mostly to prevent them being shadowed (which for
;;; reasons I don't yet understand we really don't want).
;;; The last argument is the documentation string
(defvar *count* 0 "number of times we've see this")

;;; (incf x) === (setf x (+ x 1)). The opposite is decf
(defun incr () (incf *count*))
(print *count*)
(incr)
(print *count*)
(incr)
(print *count*)

;;; Declare a global constant
(defconstant c 3e8 "speed of light")
(print c)
