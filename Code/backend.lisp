(in-package :oclcl-petalisp)

(defvar *gpu-storage-table*)
(defvar *running-in-oclcl* nil)

(defclass oclcl-backend (petalisp.core:backend)
  ((platform :initarg :platform :reader oclcl-platform)
   (device   :initarg :device   :reader oclcl-device)
   (context  :initarg :context :reader oclcl-context)
   (queue    :reader oclcl-queue)
   (kernel-cache :initform (make-hash-table) :reader oclcl-kernel-cache)))
(defmethod initialize-instance :after ((backend oclcl-backend) &key)
  (setf (slot-value backend 'queue)
        (#-opencl-2.0 eazy-opencl.host:create-command-queue
         #+opencl-2.0 eazy-opencl.host:create-command-queue-with-properties
         (oclcl-context backend)
         (oclcl-device backend))))

(defgeneric compile-kernel (backend kernel)
  (:method ((backend oclcl-backend) kernel)
    ;; Surprisingly, compiling these things isn't too slow. We don't need this message.
    ;; (format *debug-io* "~&Compiling ~s...~%" (petalisp.ir:kernel-blueprint kernel))
    (with-standard-io-syntax 
      (let* ((gpu-code (kernel->gpu-code kernel))
             (oclcl-code (gpu-code->oclcl-code gpu-code))
             (program (eazy-opencl.host:create-program-with-source
                       (oclcl-context backend)
                       (oclcl-info-program oclcl-code))))
        (eazy-opencl.host:build-program program)
        (make-gpu-kernel :program program
                         :load-instructions
                         (oclcl-info-load-instructions oclcl-code))))))

(defgeneric find-kernel (backend kernel)
  (:documentation "Find a GPU kernel that will run the code in the kernel KERNEL.")
  (:method ((backend oclcl-backend) kernel)
    (let ((blueprint (petalisp.ir:kernel-blueprint kernel)))
    (multiple-value-bind (gpu-kernel win?)
        (gethash blueprint (oclcl-kernel-cache backend))
      (if win?
          gpu-kernel
          (setf (gethash blueprint (oclcl-kernel-cache backend))
                (compile-kernel backend kernel)))))))

(defgeneric buffer-suitable-p (backend buffer)
  (:documentation "A predicate that is satisfied when the buffer BUFFER can be used by the backend BACKEND with an alien OpenCL device.")
  (:method ((backend oclcl-backend) buffer)
    (let ((storage (petalisp.ir:buffer-storage buffer)))
      (or (gpu-array-p storage)
          (subtypep (array-element-type storage) 'float)
          (every #'floatp (make-array (array-total-size storage)
                                      :displaced-to storage
                                      :element-type (array-element-type storage)))))))

(defgeneric execute-kernel (backend kernel)
  (:method ((backend oclcl-backend) kernel)
    (execute-gpu-kernel backend
                        (find-kernel backend kernel)
                        kernel)))

(defmethod petalisp.core:backend-compute ((backend oclcl-backend) (lazy-arrays list))
  (mapcar #'gpu-array->array
          (petalisp.scheduler:schedule-on-workers
           lazy-arrays
           1
           (lambda (tasks)
             (loop for task in tasks
                   for kernel = (petalisp.scheduler:task-kernel task)
                   do (execute-kernel backend kernel)))
           (constantly nil)
           (lambda (buffer)
             (let* ((dimensions (mapcar #'petalisp:range-size
                                        (petalisp:shape-ranges
                                         (petalisp.ir:buffer-shape buffer)))))
               (setf (petalisp.ir:buffer-storage buffer)
                     (make-gpu-array backend dimensions))))
           (lambda (buffer)
             (unless (null (petalisp.ir:buffer-storage buffer))
               (setf (petalisp.ir:buffer-storage buffer) nil))))))
