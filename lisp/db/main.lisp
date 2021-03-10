(load "/home/christopher/.sbclrc")
(require "cl-ppcre")
(require "cl-csv")

(defvar *query* nil "Query string")
(defvar *tokens* nil "Query tokens")
(defvar key-words #("select" "from" "where" "limit"))
(defvar *tables* (make-hash-table))

(defun segment-query ()

)

(defun tokenize-query ()
  (setq *tokens* (cl-ppcre:split "\\s+" *query*))
  (print *tokens*)
)

(defun parse-query ()
  (tokenize-query)
  (segment-query)
)

(defun load-table (name file)
  ; (print (parse-namestring file))
  (let* (
        (table-all (cl-csv:read-csv (parse-namestring file)))
        (table-data (nthcdr 1 table-all))
        (table-keys (nth 0 table-all))
    )
    (print table-keys)
    (print table-data)
  )
)


(defun main-loop ()
  (format t "~%")
  (load-table "students" (concatenate 'string
          (directory-namestring *load-pathname*) "data/students.csv"))
  (loop
    (format t "db > ")
    (finish-output)
    (setq *query* (read-line))
    ; (format t "~A~%" *query*)
    (parse-query)
  )
)

(main-loop)
