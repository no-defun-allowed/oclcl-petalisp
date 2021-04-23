(in-package :oclcl-petalisp)


(defvar *iteration-space*)
(defvar *ranges*)
(defvar *array-variables*)
(defvar *reduction-size*)
(defvar *array-id-counter*)

(defstruct gpu-code ranges code arrays reduction-size)
(defstruct oclcl-info program load-instructions)
(defstruct gpu-kernel program load-instructions)

(defun add-array (buffer)
  (let ((id (incf *array-id-counter*)))
    (setf (gethash buffer *array-variables*)
          (list (alexandria:format-symbol nil "STORAGE~a" id)
                (alexandria:format-symbol nil "SIZE~a" id)))))

(defun row-major-index (size subscripts)
  "Generate the code required to get the row-major position of the subscripts for an array of the given size.
This implentation is based on the \"possible definition\" in http://www.lispworks.com/documentation/HyperSpec/Body/f_ar_row.htm"
  (cons '+
        (maplist (lambda (x y)
                   (list* '* (first x) (rest y)))
                 subscripts
                 (loop for subscript in subscripts
                       for size-subscript from 0
                       collect `(aref ,size ,size-subscript)))))
              

(defun array-ref (array subscripts)
  (destructuring-bind (storage size)
      (gethash array *array-variables*)
    `(aref ,storage ,(row-major-index size subscripts))))

(defun stagger-operators (code)
  "Convert some code to always use 2 arguments for operators such as +, which take any number of arguments in CL, but don't in oclcl."
  (etypecase code
    (symbol code)
    (number code)
    (list
     (let ((code (mapcar #'stagger-operators code)))
       (destructuring-bind (function &rest arguments) code
         (case function
           ((+ - * /)
            (case (length arguments)
              (0 (ecase function
                   ((member / -) (error "~s is not defined for zero arguments" function))
                   ((+) 0)
                   ((*) 1)))
              (1 (if (eql function '/)
                     `(/ 1.0 ,(first arguments))
                     (first arguments)))
              (2 (cons function arguments))
              (t (reduce (lambda (a b) (list function a b))
                         arguments :from-end t))))
           ((1+)
            (destructuring-bind (1+ number) code
              (declare (ignore 1+))
              `(+ ,number 1)))
           ((1-)
            (destructuring-bind (1- number) code
              (declare (ignore 1-))
              `(- ,number 1)))
           (t
            (cons function arguments))))))))

(defun kernel->gpu-code (kernel)
  (let* ((compiled-instructions '())
         (*iteration-space* (petalisp.ir:kernel-iteration-space kernel))
         (*ranges* (loop for range in (petalisp:shape-ranges *iteration-space*)
                         for index from 0
                         collect (alexandria:format-symbol nil "RANGE~d" index)))
         (*array-variables* (make-hash-table))
         (*reduction-size* (make-symbol "REDUCTION-SIZE"))
         (*array-id-counter* 0))
    (add-array :output)
    (petalisp.ir:map-kernel-inputs #'add-array kernel)
    (petalisp.ir:map-kernel-store-instructions
     (lambda (instruction)
       (push (compile-store-instruction instruction) compiled-instructions))
     kernel)
    (let ((arrays (alexandria:hash-table-alist *array-variables*)))
      ;; Ensure that the output buffer is first
      (make-gpu-code :arrays (cons (find :output arrays :key #'first)
                                   (remove :output arrays :key #'first))
                     :code `(let ,(loop for range in *ranges*
                                        for index from 0
                                        collect `(,range
                                                  (oclcl.lang:to-int
                                                   (oclcl.lang:get-global-id ,index))))
                              ,@compiled-instructions)
                     :ranges (rest *ranges*)
                     :reduction-size *reduction-size*))))

(defun gpu-code->oclcl-code (gpu-code)
  (let ((oclcl:*program* (oclcl:make-program :name "Petalisp program")))
    (oclcl:program-define-function oclcl:*program*
                                   'kernel
                                   'oclcl:void
                                   (loop for (id storage size) in (gpu-code-arrays gpu-code)
                                         collect (list storage 'oclcl:float*)
                                         collect (list size 'oclcl:int*))
                                   (list (stagger-operators (gpu-code-code gpu-code))))
    (make-oclcl-info :program (oclcl:compile-program oclcl:*program*)
                     :load-instructions (remove :output (mapcar #'first (gpu-code-arrays gpu-code))))))
                                         

(defmacro with-input ((instruction input) &body body)
  `(progn
     (assert (zerop (car ,input)) () "Cannot compile multiple values")
     (let ((,instruction (cdr ,input)))
       ,@body)))

(defun transform-by-instruction (input instruction)
  (petalisp:transform-sequence
   input
   (petalisp.ir:instruction-transformation instruction)))

(defun compile-normal-store (instruction store-instruction)
  `(set ,(array-ref :output (transform-by-instruction *ranges* store-instruction))
        ,(compile-inner-instruction instruction)))


(defun compile-store-instruction (instruction)
  "Compile a STORE-INSTRUCTION that may contain a reduction."
  (let ((input (first (petalisp.ir:instruction-inputs instruction))))
    (with-input (input input)
      (compile-normal-store input instruction))))

(defgeneric compile-inner-instruction (instruction)
  (:method ((load petalisp.ir:load-instruction))
    (array-ref (petalisp.ir:load-instruction-buffer load)
               (transform-by-instruction *ranges* load)))
  (:method ((call petalisp.ir:call-instruction))
    `(,(function-name (petalisp.ir:call-instruction-operator call))
      ,@(loop for input in (petalisp.ir:instruction-inputs call)
              collect (with-input (input input)
                        (compile-inner-instruction input)))))
  (:method ((iref petalisp.ir:iref-instruction))
    `(oclcl.lang:to-float ,(first (transform-by-instruction *ranges* iref)))))

(defun function-name (function)
  (etypecase function
    (symbol
     (if (eq (symbol-package function)
             (find-package "PETALISP.TYPE-INFERENCE"))
         (let ((name (symbol-name function)))
           (cond
             ((alexandria:starts-with-subseq "SHORT-FLOAT" name)
              (intern (subseq name 11) "CL"))
             (t (error "unknown type-inference symbol ~s" function))))
         function))
    (function
     (multiple-value-bind (expression closure? name)
         (function-lambda-expression function)
       (declare (ignore expression closure?))
       (assert (not (null name)))
       (function-name name)))))
