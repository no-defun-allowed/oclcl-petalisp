(in-package :oclcl-petalisp)

(defclass oclcl-backend (petalisp.core:backend)
  ((platform :initarg :platform :reader oclcl-platform)
   (device   :initarg :device   :reader oclcl-device)
   (context  :initarg :context :reader oclcl-context)
   (queue    :reader oclcl-queue)
   (kernel-cache :initform (make-hash-table) :reader oclcl-kernel-cache)))
(defmethod initialize-instance :after ((backend oclcl-backend) &key)
  (setf (slot-value backend 'queue)
        (eazy-opencl.host:create-command-queue
         (oclcl-context backend)
         (oclcl-device backend)
         0)))

(defgeneric compile-kernel (backend kernel)
  (:method ((backend oclcl-backend) kernel)
    (let* ((gpu-code (kernel->gpu-code kernel))
           (oclcl-code (gpu-code->oclcl-code gpu-code))
           (program (eazy-opencl.host:create-program-with-source
                    (oclcl-context backend)
                    (oclcl-info-program oclcl-code))))
      (eazy-opencl.host:build-program program)
      (make-gpu-kernel :program program
                       :load-instructions
                       (oclcl-info-load-instructions oclcl-code)))))

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

(defmethod petalisp.core:compute-immediates ((lazy-arrays list) (backend oclcl-backend))
  (petalisp.scheduler:schedule-on-workers
   lazy-arrays
   1
   (lambda (tasks)
     (loop for task in tasks
           for kernel = (petalisp.scheduler:task-kernel task)
           do (execute-kernel backend kernel)))
   (constantly nil)
   (lambda (buffer)
     (setf (petalisp.ir:buffer-storage buffer)
           (make-gpu-array backend
                           (mapcar #'petalisp:range-size
                                   (petalisp:shape-ranges
                                    (petalisp.ir:buffer-shape buffer))))))
   (lambda (buffer)
     (unless (null (petalisp.ir:buffer-storage buffer))
       (setf (petalisp.ir:buffer-storage buffer) nil)))))
(defmethod petalisp.core:lisp-datum-from-immediate ((gpu-array gpu-array))
  (gpu-array->array gpu-array)) 
