(in-package :oclcl-petalisp)

(defvar *float-size* (cffi:foreign-type-size :float))
(defvar *int-size*   (cffi:foreign-type-size :int))
(defstruct (gpu-array (:constructor %make-gpu-array))
  backend storage gpu-dimensions dimensions)

(defun make-gpu-array (backend dimensions &optional foreign-storage)
  #-oclcl-petalisp/do-not-zero
  (when (and (null foreign-storage)
             (not (null (gethash dimensions (oclcl-cached-memory backend)))))
    (return-from make-gpu-array
      (pop (gethash dimensions (oclcl-cached-memory backend)))))
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
    (%make-gpu-array :backend backend
                     :storage storage-device
                     :gpu-dimensions dimensions-device
                     :dimensions dimensions)))

(defun recycle-gpu-array (backend gpu-array)
  (push gpu-array (gethash (gpu-array-dimensions gpu-array)
                           (oclcl-cached-memory backend))))

(defun array->gpu-array (backend array)
  (cffi:with-foreign-object (foreign-memory :float (array-total-size array))
    (dotimes (index (array-total-size array))
      (setf (cffi:mem-aref foreign-memory :float index)
            (coerce (row-major-aref array index) 'single-float)))
    (make-gpu-array backend (array-dimensions array) foreign-memory)))
    

(defun gpu-array->array (gpu-array)
  (let ((backend (gpu-array-backend gpu-array))
        (full-size (reduce #'* (gpu-array-dimensions gpu-array)))
        (array (make-array (gpu-array-dimensions gpu-array) :element-type 'single-float)))
    (cffi:with-foreign-object (foreign-memory :float full-size)
      (%ocl:enqueue-read-buffer (oclcl-queue backend)
                                (gpu-array-storage gpu-array)
                                %ocl:true 0 (* *float-size* full-size)
                                foreign-memory
                                0 (cffi:null-pointer) (cffi:null-pointer))
      (%ocl:finish (oclcl-queue backend))
      (dotimes (index full-size)
        (setf (row-major-aref array index)
              (cffi:mem-aref foreign-memory :float index)))
      array)))

(defmethod print-object ((gpu-array gpu-array) stream)
  (print-unreadable-object (gpu-array stream :type t :identity t)
    (format stream "~_:Dimensions ~s ~_:Storage ~s"
            (gpu-array-dimensions gpu-array)
            (gpu-array->array gpu-array))))
