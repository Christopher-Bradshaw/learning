(print "First example")
(let ((x 10))
  ;; When we do this, x is evaluated to 10
  (print x)
  ;; apostrophe is shorthand for the quote function
  (print 'x)
  (print (quote x))
)

(print "Second example")
(let ((x 10))
  ;; These two get evaluated
  (print x)
  (print (+ x 2))
  ;; This one doesn't
  (print '(+ x 2))
)
