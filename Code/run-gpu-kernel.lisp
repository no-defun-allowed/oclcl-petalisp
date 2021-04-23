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

(defun set-kernel-array (kernel id gpu-array)
  (let ((position (+ 2 (* id 2))))
    (eazy-opencl.host:set-kernel-arg kernel position
                                     (gpu-array-storage gpu-array)
                                     '%ocl:mem)
    (eazy-opencl.host:set-kernel-arg kernel (1+ position)
                                     (gpu-array-gpu-dimensions gpu-array)
                                     '%ocl:mem)))

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
      (loop for load-buffer in (petalisp-kernel-inputs kernel)
            for storage = (petalisp.ir:buffer-storage load-buffer)
            for gpu-array = (if (arrayp storage)
                                (array->gpu-array backend storage)
                                storage)
            for id from 0
            do (set-kernel-array opencl-kernel id gpu-array))
      (let ((iteration-ranges ranges))
        (cffi:with-foreign-array (work '%ocl:size-t
                                       (or (mapcar #'petalisp:range-size iteration-ranges)
                                           (list 1)))
          (%ocl:enqueue-nd-range-kernel queue opencl-kernel
                                        (max 1 (length iteration-ranges)) (cffi:null-pointer)
                                        work (cffi:null-pointer)
                                        0 (cffi:null-pointer) (cffi:null-pointer)))))))
