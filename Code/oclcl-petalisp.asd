(asdf:defsystem :oclcl-petalisp
  :depends-on (:petalisp :eazy-opencl :oclcl)
  :serial t
  :components ((:file "package")
               (:file "device-picker")
               (:file "code-generator")
               (:file "gpu-array")
               (:file "primops")
               (:file "backend")
               (:file "run-gpu-kernel")))
