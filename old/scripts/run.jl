if size(ARGS, 1) == 4
  trailfile=ARGS[1]
  testfile=ARGS[2]
  outputfolder=ARGS[3]
  modelfile=ARGS[4]
  options = ""
else if size(ARGS, 1) == 5
  trailfile=ARGS[1]
  testfile=ARGS[2]
  outputfolder=ARGS[3]
  modelfile=ARGS[4]
  options = ARGS[5]
else if size(ARGS, 1) == 1 && ARGS[1] = "auto"
  trailfile, testfile, outputfolder, modelfile =
  "data/wells/wells_trails",
  "data/wells/wells_tests",
  "data/output/curr",
  "wells_model"
else
  error("bad command line arguments")
end
runSVM(trailfile, testfile, outputfolder, modelfile, options=options)


# runSVM("data/wells/wells_trails",
# 	"data/wells/wells_tests",
# 	"data/output/curr",
# 	"wells_model",
# 	options = "svm_type=EPSILON_SVR, C=100.0, gamma=1.0, eps=0.03125")	