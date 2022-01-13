#=
!NOTES: The NUTS kernel and stepsize adaption are adapted from the excellent package
  https://github.com/tpapp/DynamicHMC.jl

The DynamicHMC.jl package is licensed under the MIT "Expat" License:

> Copyright (c) 2020: Tamas K. Papp.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
>
=#

############################################################################################
# Constants
"Maximum number of iterations"
const MAX_DIRECTIONS_DEPTH = 32

"Default maximum depth for trees."
const DEFAULT_MAX_TREE_DEPTH = 10

############################################################################################
#Include
include("treebuilding.jl")
include("kernel.jl")
include("config.jl")
include("diagnostics.jl")
include("postprocessing.jl")

############################################################################################
#export
