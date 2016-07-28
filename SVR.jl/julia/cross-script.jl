l = size(ARGS, 1)
trailfile = ARGS[1]
nr_fold = parse(ARGS[2])
options = ""
if l!=2 && l!=3
  println("please enter trainingfile nr_folds svmoptions(optionalinput) in the following fomat:")
  println("trainingfile nr_folds option1=val1,option2=val3,...")
else
  if l == 3
    options = ARGS[3]
  end
  include("src.jl")
  do_cross_validation(trailfile, nr_fold, options=options)
end


