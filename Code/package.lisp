(defpackage :oclcl-petalisp
  (:use :cl)
  (:export #:list-platforms
           #:choose-device))

(defpackage :oclcl-petalisp-primops
  (:use)
  (:export #:is-equal #:select #:define-primop))
