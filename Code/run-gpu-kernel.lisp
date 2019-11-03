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

(defgeneric execute-gpu-kernel (backend gpu-kernel kernel)
  (:method ((backend oclcl-backend) gpu-kernel kernel)
    (let* ((queue (eazy-opencl.host:create-command-queue
                   (oclcl-context backend)
                   (oclcl-device backend)))
           (opencl-kernel (eazy-opencl.host:create-kernel
                           (gpu-kernel-program gpu-kernel)
                           "oclcl_petalisp_kernel"))
           (gpu-array (petalisp.ir:buffer-storage
                       (petalisp-kernel-output-buffer kernel)))
           (load-instructions (petalisp.ir::kernel-load-instructions kernel)))
        (eazy-opencl.host:set-kernel-arg opencl-kernel 0
                                         (gpu-array-storage gpu-array)
                                         '%ocl:mem)
        (eazy-opencl.host:set-kernel-arg opencl-kernel 1
                                         (gpu-array-gpu-dimensions gpu-array)
                                         '%ocl:mem)
        (loop for load-instruction-number in (gpu-kernel-load-instructions gpu-kernel)
              for load-instruction = (find load-instruction-number
                                           load-instructions
                                           :key #'petalisp.ir:instruction-number)
              for load-buffer = (petalisp.ir:buffer-storage
                                 (petalisp.ir:load-instruction-buffer
                                  load-instruction))
              for storage-position from 2 by 2
              for size-position    = (1+ storage-position)
              do (when (arrayp load-buffer)
                   (setf load-buffer (array->gpu-array backend load-buffer)))
                 (eazy-opencl.host:set-kernel-arg opencl-kernel storage-position
                                                  (gpu-array-storage load-buffer)
                                                  '%ocl:mem)
                 (eazy-opencl.host:set-kernel-arg opencl-kernel size-position
                                                  (gpu-array-gpu-dimensions load-buffer)
                                                  '%ocl:mem))
        (let ((iteration-ranges (rest (petalisp:shape-ranges
                                       (petalisp.ir:kernel-iteration-space
                                        kernel)))))
          (cffi:with-foreign-array (work '%ocl:size-t
                                         (mapcar #'petalisp:range-size iteration-ranges))
            (%ocl:enqueue-nd-range-kernel queue opencl-kernel
                                          (length iteration-ranges) (cffi:null-pointer)
                                          work (cffi:null-pointer)
                                          0 (cffi:null-pointer) (cffi:null-pointer))
            (%ocl:finish queue))))))
                                          
                                                  
