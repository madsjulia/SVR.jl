using JLD
s = Base.source_path()
s = split(s, "/")
s = s[1:end-1]
currdir = join(s, "/")
dirs = readdir(string(currdir, "/data_svr"))
dirs = dirs[1001:2000]
tempd = dirs[1]
d = load(string("data_svr/", tempd))

full = Array{Float32}(size(dirs, 1)*11, size(d[collect(keys(d))[2]]', 2)+1)

for i=1:size(dirs, 1)
	print(".")
	dir = dirs[i]
	d = load(string("data_svr/", dir))
	features = d[collect(keys(d))[2]]'
	obs = d[collect(keys(d))[1]][1:11]
	t = Array{Float64}(size(obs, 1), size(features, 2)+1)
	t[:, 1] = obs

	for j=1:size(t, 1)
	   t[j, 2:end] = features
	end

	full[size(obs, 1)*(i-1) + 1: size(obs, 1)*(i), :] = t
end

save("predict_data.jld", "data", full)
d = load("predict_data.jld")
data = d["data"]
writecsv("predict.csv",data)

