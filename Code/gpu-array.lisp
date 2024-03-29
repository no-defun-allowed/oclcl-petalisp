(in-package :oclcl-petalisp)

(defvar *float-size* (cffi:foreign-type-size :float))
(defvar *int-size*   (cffi:foreign-type-size :int))
(defstruct (gpu-array (:constructor %make-gpu-array))
  backend storage gpu-dimensions dimensions)

(defstruct allocation-meters
  (allocated 0)
  (reused 0))

(defun make-gpu-array (backend dimensions &key foreign-storage)
  ;; Try to grab unused arrays first.
  #-oclcl-petalisp/do-not-zero
  (unless (null (gethash dimensions (oclcl-cached-memory backend)))
    (incf (allocation-meters-reused (oclcl-allocation-meters backend)))
    (let ((gpu-array
            (pop (gethash dimensions (oclcl-cached-memory backend)))))
      (push gpu-array *allocated*)
      (unless (null foreign-storage)
        (let ((queue (oclcl-queue backend))
              (size  (reduce #'* dimensions))
              (storage-device (gpu-array-storage gpu-array)))
          (%ocl:enqueue-write-buffer queue storage-device %ocl:true
                                     0 (* *float-size* size) foreign-storage
                                     0 (cffi:null-pointer) (cffi:null-pointer))
          (%ocl:finish queue)))
      (return-from make-gpu-array gpu-array)))
  (print dimensions)
  (incf (allocation-meters-allocated (oclcl-allocation-meters backend)))
  (let* ((size (reduce #'* dimensions))
         (context (oclcl-context backend))
         (queue (oclcl-queue backend))
         (dimensions-size (* *int-size* (max 1 (length dimensions))))
         (dimensions-device
           (eazy-opencl.host:create-buffer context
                                           :mem-read-only
                                           dimensions-size))
         (storage-device
           (eazy-opencl.host:create-buffer context
                                           :mem-read-write
                                           (* *float-size* size))))
    (cffi:with-foreign-array (foreign-dimensions :int dimensions)
      ;; Write foreign-dimensions to dimensions-device
      (%ocl:enqueue-write-buffer queue
                                 dimensions-device
                                 %ocl:true
                                 0 dimensions-size
                                 foreign-dimensions
                                 0 (cffi:null-pointer) (cffi:null-pointer))
      ;; Write foreign-storage if we have any to storage-buffer
      (if (not (null foreign-storage))
          (%ocl:enqueue-write-buffer queue storage-device %ocl:true
                                     0 (* *float-size* size) foreign-storage
                                     0 (cffi:null-pointer) (cffi:null-pointer))
          #-oclcl-petalisp/do-not-zero
          (cffi:with-foreign-array (foreign-storage :float (make-array size :initial-element 0.0s0))
            (%ocl:enqueue-write-buffer queue storage-device %ocl:true
                                       0 (* *float-size* size) foreign-storage
                                       0 (cffi:null-pointer) (cffi:null-pointer))))
      (%ocl:finish queue))
    (let ((gpu-array
            (%make-gpu-array :backend backend
                             :storage storage-device
                             :gpu-dimensions dimensions-device
                             :dimensions dimensions)))
      (push gpu-array *allocated*)
      gpu-array)))

(defun recycle-gpu-array (backend gpu-array)
  (push gpu-array (gethash (gpu-array-dimensions gpu-array)
                           (oclcl-cached-memory backend))))

(defun array->gpu-array (backend array)
  #+sbcl
  (when (typep array '(simple-array single-float))
    (return-from array->gpu-array
      (cffi:with-pointer-to-vector-data (p (sb-ext:array-storage-vector array))
        (make-gpu-array backend (array-dimensions array) :foreign-storage p))))
  (cffi:with-foreign-object (foreign-memory :float (array-total-size array))
    (dotimes (index (array-total-size array))
      (setf (cffi:mem-aref foreign-memory :float index)
            (coerce (row-major-aref array index) 'single-float)))
    (make-gpu-array backend (array-dimensions array) :foreign-storage foreign-memory)))

(defun gpu-array->array (gpu-array)
  (let ((backend (gpu-array-backend gpu-array))
        (full-size (reduce #'* (gpu-array-dimensions gpu-array)))
        (array (make-array (gpu-array-dimensions gpu-array) :element-type 'single-float)))
    (symbol-macrolet ((do-the-read
                          (%ocl:enqueue-read-buffer
                           (oclcl-queue backend)
                           (gpu-array-storage gpu-array)
                           %ocl:true 0 (* *float-size* full-size)
                           foreign-memory
                           0 (cffi:null-pointer) (cffi:null-pointer))))
      #+sbcl
      (cffi:with-pointer-to-vector-data (foreign-memory (sb-ext:array-storage-vector array))
        do-the-read)
      #-sbcl
      (cffi:with-foreign-object (foreign-memory :float full-size)
        do-the-read
        (%ocl:finish (oclcl-queue backend))
        (dotimes (index full-size)
          (setf (row-major-aref array index)
                (cffi:mem-aref foreign-memory :float index))))
      array)))

(defmethod print-object ((gpu-array gpu-array) stream)
  (print-unreadable-object (gpu-array stream :type t :identity t)
    (format stream "~_:Dimensions ~s ~_:Storage ~s"
            (gpu-array-dimensions gpu-array)
            (gpu-array->array gpu-array))))

(defvar *allocated*)
(defmacro with-allocation-pool ((backend) &body body)
  `(let ((*allocated* '()))
     (multiple-value-prog1
         (progn ,@body)
       (dolist (a *allocated*)
         (recycle-gpu-array ,backend a)))))
