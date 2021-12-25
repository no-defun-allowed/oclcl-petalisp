(in-package :oclcl-petalisp)

(defvar *primops* '((oclcl-petalisp-primops:select
                     (test sub alt)
                     (if (/= test 0.0)
                         (return sub)
                         (return alt)))
                    (oclcl-petalisp-primops:is-equal
                     (x y)
                     (if (= x y)
                         (return 1.0)
                         (return 0.0)))))

(defvar *patch-cl-functions* '((max . oclcl:fmax)
                               (min . oclcl:fmin)))

(defun add-primops ()
  (loop for (name lambda-list code) in *primops*
        do (oclcl:program-define-function oclcl:*program*
                                          name
                                          'oclcl:float
                                          (loop for parameter in lambda-list
                                                collect (list parameter 'oclcl:float))
                                          (list code))))

(defmacro oclcl-petalisp-primops:define-primop (name arguments code)
  `(progn
     (setf *primops*
           (remove ',name *primops* :key #'first))
     (push (list ',name ',arguments ',code) *primops*)
     ',name))

(declaim (inline oclcl-petalisp-primops:is-equal
                 oclcl-petalisp-primops:select))
(defun oclcl-petalisp-primops:is-equal (x y)
  (if (= x y) 1.0 0.0))
(defun oclcl-petalisp-primops:select (test then else)
  (if (zerop test) else then))
