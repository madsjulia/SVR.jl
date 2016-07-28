using Distributions


ar = [-10, 10]
br = [-10, 10]
cr = [-5, 5]
nr = [-3, 3]
t = [1,2,3,4,5]

function f_cvs(ar, br, cr, nr, t)
              file = open("results.csv" , "w")
              write(file, "f,   a,   b,   c,   n,   t\n")

              i = 0
              while i < 100
              a_d =DiscreteUniform((ar[1]), (ar[end]))
             b_d = DiscreteUniform((br[1]), (br[end]))
              c_d = DiscreteUniform((cr[1]), (cr[end]))
              n_d = DiscreteUniform((nr[1]), (nr[end]))
              a = rand(a_d)
              b = rand(b_d)
              c = rand(c_d)
              n = rand(n_d)
              for (j, ts) in enumerate(t)
              f = (a*float(ts)^(n))+(b*ts)+c
              write(file, "$f,   $a,     $b,    $c,    $n,    $ts\n")
              print(".")
              end
              i += 1
              end
              close(file)
              end
              
              f_cvs(ar, br, cr, nr, t)
