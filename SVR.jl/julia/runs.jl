argv1 = [ "svm-train", "-s", "3", "data/wells/wells_trails", "data/wells/wells_model" ]
argv2 = [ "svm-predict", "data/wells/wells_tests", "data/wells/wells_model", "data/wells/wells_output" ]
ccall((:main, "/n/srv/vessg/Downloads/libsvm-3.21/svm-train.so"), Int32, (Int32, Ptr{Ptr{UInt8}}), length(argv1), argv1)
ccall((:main, "/n/srv/vessg/Downloads/libsvm-3.21/svm-predict.so"), Int32, (Int32, Ptr{Ptr{UInt8}}), length(argv2), argv2)
