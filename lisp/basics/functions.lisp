;;; defun funcname (args list) (body)
(defun add(a b) (+ a b))
(format t "1 + 2 = ~D~%" (add 1 2))


;;; optional arguments
(defun area(height &optional width)
  (cond
    (width (* height width))
    (T (* height height))
  )
)

(print (list (area 5 4) (area 5)))

;;; optional arguments, with default
(defun area-with-default(height &optional (width 10))
  (cond
    (width (* height width))
    (T (* height height))
  )
)
(print (list (area-with-default 5 4) (area-with-default 5)))

;;; optional arguments, with default as other arg
(defun area-with-arg-default(height &optional (width height))
  (* width height)
)
(print (list (area-with-arg-default 5 4) (area-with-arg-default 5)))


;;; keyword arguments
(defun add-and-print(a b &key also-print)
  (if also-print (print (+ a b)))
  (+ a b)
)

(print (add-and-print 1 2))
(add-and-print 1 2 :also-print T)
