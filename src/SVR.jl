__precompile__()

"""
SVR.jl: Support Vector Machine Regression
"""
module SVR

import Base
import Libdl
import DelimitedFiles
import DocumentFunction
import Statistics

include("SVRconstants.jl")
include("SVRlib.jl")
include("SVRfunctions.jl")

end