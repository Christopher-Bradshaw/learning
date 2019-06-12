(defun fibonacci (n)
  (cond
    ((eql n 1) 1)
    ((eql n 2) 1)
    (t (+ (fibonacci (- n 1)) (fibonacci (- n 2))))
  )
)

(print (fibonacci 3))
