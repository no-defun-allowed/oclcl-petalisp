(asdf:defsystem :oclcl-petalisp
  :depends-on (:petalisp :eazy-opencl :oclcl)
  :components ((:file "package")
               (:file "device-picker")
               (:file "backend")
               (:file "code-generator")
               (:file "gpu-array")
               (:file "run-gpu-kernel")))
