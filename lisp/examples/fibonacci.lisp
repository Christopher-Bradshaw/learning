;;;; File level comment

;;; fibonacci generates the x'th fib number
(defun fibonacci (x)
  (cond
    ((eql x 1) 1)
    ((eql x 2) 1)
    ;; Always evalutes to True (if we get here)
    (T (+ (fibonacci (- x 1)) (fibonacci (- x 2))))
  )
)

;;; The variable x is scoped to the let block
(let ((x 20))
    (format t "fibonacci(~D) = ~D~%" x (fibonacci x))
)

;;; So x is not defined down here
