l = 1000000
barlen = 50
modby = Int(l/barlen)
count = 0
for i=1:l
    if i%modby==0
	print("\r")
	count+=1
	print("|")
	for j=1:barlen
	    if j<=count
		print("=")
	    else
		print(" ")
	    end
	end
	print("|")
    end
end