import PyCall
path_to_use_for_grid = dirname(Base.source_path())

function grid(num_w, ssh_w, telnet_w, trialfile, options)
	trialfile = abspath(trialfile)
	options = split(options, " ")
	unshift!(PyCall.PyVector(PyCall.pyimport("sys")["path"]), "")
	p = pwd()
	cd(path_to_use_for_grid)
	@PyCall.pyimport gridregressionjulia
	gridregressionjulia.set_for_parallel(num_w, ssh_w, telnet_w)
	best_mse, best_param = gridregressionjulia.find_parameters(trialfile, options)
	cd(p)
	return best_mse, best_param
end