function convertSVM(infile, outfile)
  fin = open(infile, "r")
  fout = open(outfile, "a")
  
  while true
    a = readline(fin)
    if a == ""
      break
    end
    a = split(a, ",")
    a = map(x->Float64(parse(x)), a)
    printout(fout, a)
  end
  
  close(fin)
  close(fout)
  
end

function printout(fout, a)
  @printf(fout, "%f ", a[1])
  for i=2:size(a, 1)
    if a[i] != 0.0
      @printf(fout, " %d:%f", i-1, a[i])
    end
  end
  @printf(fout, "\n")
end
