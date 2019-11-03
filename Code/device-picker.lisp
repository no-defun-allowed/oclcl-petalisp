(in-package :oclcl-petalisp)

(defun list-platforms (&optional (*standard-output* *standard-output*))
  (loop for platform in (eazy-opencl.host:get-platform-ids)
        for n from 0
        do (format t "Platform ~1d: ~a from ~a, ~a~%"
		   n
		   (eazy-opencl.host:get-platform-info platform :platform-name)
		   (eazy-opencl.host:get-platform-info platform :platform-vendor)
		   (eazy-opencl.host:get-platform-info platform :platform-version))
           (list-devices platform n)))

(defun list-devices (platform n)
  (loop for device in (eazy-opencl.host:get-device-ids platform :device-type-default)
        for dn from 0
        do (format t "Device ~2d,~2d: ~a, ~4d MHz, ~4d MiB memory~%"
		   n dn
		   (eazy-opencl.host:get-device-info device :device-name)
		   (eazy-opencl.host:get-device-info device :device-max-clock-frequency)
		   (round (eazy-opencl.host:get-device-info device :device-global-mem-size)
			  #.(expt 2 20)))))

(defun choose-device (platform-n)
  (let* ((platform (elt (eazy-opencl.host:get-platform-ids) platform-n))
         (devices  (eazy-opencl.host:get-device-ids platform :device-type-default))
         (context  (eazy-opencl.host:create-context devices :context-platform platform))) 
    (make-instance 'oclcl-backend
                   :platform platform
                   :device (first devices)
                   :context context)))
