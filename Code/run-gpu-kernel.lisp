(in-package :oclcl-petalisp)

(defun petalisp-kernel-output-buffer (kernel)
  (let ((buffers '()))
    (petalisp.ir:map-kernel-outputs (lambda (buffer)
                                      (push buffer buffers))
                                    kernel)
    (assert (= 1 (length buffers)))
    (first buffers)))

(defun petalisp-kernel-inputs (kernel)
  (let ((buffers '()))
    (petalisp.ir:map-kernel-inputs
     (lambda (buffer) (push buffer buffers))
     kernel)
    buffers))

(defun %set-kernel-mem-arg (kernel index value)
  (cffi:with-foreign-object (p '%ocl:mem)
    (setf (cffi:mem-ref p '%ocl:mem) value)
    (%ocl/e:set-kernel-arg kernel index #.(cffi:foreign-type-size '%ocl:mem) p)))

(defun set-kernel-array (kernel id gpu-array)
  (let ((position (+ 3 (* id 2))))
    (%set-kernel-mem-arg kernel position
                         (gpu-array-storage gpu-array))
    (%set-kernel-mem-arg kernel (1+ position)
                         (gpu-array-gpu-dimensions gpu-array))))

(defun set-iteration-limits (backend kernel iteration-ranges)
  (cffi:with-foreign-array (limits :int
                                   (mapcar #'petalisp:range-size
                                           iteration-ranges))
    (let* ((size (* *int-size* (max 1 (length iteration-ranges))))
           (buffer (eazy-opencl.host:create-buffer (oclcl-context backend)
                                                   :mem-read-only size)))
      (unless (null iteration-ranges)
        (%ocl:enqueue-write-buffer (oclcl-queue backend)
                                   buffer
                                   %ocl:true
                                   0 size limits
                                   0 (cffi:null-pointer) (cffi:null-pointer))
        (%ocl:finish (oclcl-queue backend)))
      (%set-kernel-mem-arg kernel 0 buffer))))

(defgeneric execute-gpu-kernel (backend gpu-kernel kernel)
  (:method ((backend oclcl-backend) gpu-kernel kernel)
    (let* ((queue (oclcl-queue backend))
           (opencl-kernel (eazy-opencl.host:create-kernel
                           (gpu-kernel-program gpu-kernel)
                           "oclcl_petalisp_kernel"))
           (gpu-array
             (petalisp.ir:buffer-storage
              (petalisp-kernel-output-buffer kernel)))
           (ranges
             (petalisp:shape-ranges
              (petalisp.ir:kernel-iteration-space kernel))))
      (set-kernel-array opencl-kernel -1 gpu-array)
      (set-iteration-limits backend opencl-kernel ranges)
      (loop for load-buffer in (petalisp-kernel-inputs kernel)
            for storage = (petalisp.ir:buffer-storage load-buffer)
            for gpu-array = (if (arrayp storage)
                                (array->gpu-array backend storage)
                                storage)
            for id from 0
            do (set-kernel-array opencl-kernel id gpu-array))
      (let ((ranges (loop for range in ranges
                          collect (ceiling (petalisp:range-size range)
                                           (gpu-kernel-chunk-size gpu-kernel)))))
        (cffi:with-foreign-array (work '%ocl:size-t
                                       (if (null ranges) (list 1) ranges))
          (%ocl:enqueue-nd-range-kernel queue opencl-kernel
                                        (max 1 (length ranges)) (cffi:null-pointer)
                                        work (cffi:null-pointer)
                                        0 (cffi:null-pointer) (cffi:null-pointer)))))))
