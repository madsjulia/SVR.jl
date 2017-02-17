height=10
for k = 1:Base.tty_size()[2]
	sleep(0.025)
	for i=1:Base.tty_size()[1]
		if i==1
			print("--\n")
		elseif i==Base.tty_size()[1]
			for j=1:Base.tty_size()[2]
				if j==Base.tty_size()[2]-k+1 || j==Base.tty_size()[2]-k+2
					print("|")
				else
					print("_")
				end
			end
		else
			for j=1:Base.tty_size()[2]-k
				print(" ")
			end
			if i>(Base.tty_size()[1] - height)
				print("||")
				print("\n")
			else
				print("\n")
			end
		end
	end
end
