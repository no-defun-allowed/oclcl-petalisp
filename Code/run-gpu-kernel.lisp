(in-package :oclcl-petalisp)

(defun petalisp-kernel-output-buffer (kernel)
  (let ((buffers '()))
    (petalisp.ir:map-kernel-outputs (lambda (buffer)
                                      (push buffer buffers))
                                    kernel)
    (assert (= 1 (length buffers)))
    (first buffers)))
(defun petalisp-kernel-load-instructions (kernel)
  (let ((instructions '()))
    (petalisp.ir:map-kernel-load-instructions
     (lambda (instructions)
       (push instructions instructions))
     kernel)
    (sort instructions #'< :key #'petalisp.ir:instruction-number)))

(defun set-kernel-array (kernel id gpu-array)
  (let ((position (+ 3 (* id 2))))
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
             (gethash (petalisp.ir:buffer-storage
                       (petalisp-kernel-output-buffer kernel))
                       *gpu-storage-table*))
           (load-instructions (petalisp.ir::kernel-load-instructions kernel))
           (ranges
             (petalisp:shape-ranges
              (petalisp.ir:kernel-iteration-space kernel))))
      (eazy-opencl.host:set-kernel-arg opencl-kernel 0
                                       (petalisp:range-size (first ranges))
                                       '%ocl:int)
      (set-kernel-array opencl-kernel -1 gpu-array)
      (loop for load-instruction-number in (gpu-kernel-load-instructions gpu-kernel)
            for load-instruction = (find load-instruction-number
                                         load-instructions
                                         :key #'petalisp.ir:instruction-number)
            for load-buffer = (petalisp.ir:load-instruction-buffer
                               load-instruction)
            for gpu-array = (or (gethash (petalisp.ir:buffer-storage load-buffer)
                                         *gpu-storage-table*)
                                (array->gpu-array
                                 backend
                                 (petalisp.ir:buffer-storage load-buffer)))
            for id from 0
            do (set-kernel-array opencl-kernel id gpu-array))
      (let ((iteration-ranges (rest ranges)))
        (cffi:with-foreign-array (work '%ocl:size-t
                                       (or (mapcar #'petalisp:range-size iteration-ranges)
                                           (list 1)))
          (%ocl:enqueue-nd-range-kernel queue opencl-kernel
                                        (max 1 (length iteration-ranges)) (cffi:null-pointer)
                                        work (cffi:null-pointer)
                                        0 (cffi:null-pointer) (cffi:null-pointer)))))))
