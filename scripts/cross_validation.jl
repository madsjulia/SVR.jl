function do_cross_validation(pprob, pparam, nr_fold)
  prob = unsafe_load(pprob)
  param = unsafe_load(pparam)
  total_correct = 0
  total_error = sumv = sumy = sumvv = sumyy = sumvy = 0.0
  target = Array{Float64}(prob.l)

  ccall((:svm_cross_validation, svmlib), Void, (Ptr{svm_problem}, Ptr{svm_parameter}, Cint, Ptr{Float64}), pprob, pparam, nr_fold, pointer(target))

  if param.svm_type == EPSILON_SVR || param.svm_type == NU_SVR
    for i=1:prob.l
      y = prob.y[i]
      v = target[i]
      sumv+=v
      sumy+=y
      sumvv+v*v
      sumyy+=y*y
      sumvy+=v*y
    end
    @printf("Cross Validation Mean squared error = %g\n",total_error/prob.l)
    @printf("Cross Validation Squared correlation coefficient = %g\n",
	    ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
	    ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
	    )
  else
    for i=1:prob.l
      if target[i] = prob.y[i]
	total_correct+=1
	@printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l)
      end
    end
  end

end