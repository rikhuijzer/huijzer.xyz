### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 87efa960-4dbb-42c3-9ad5-7189f58beeb5
using PlutoUI: with_terminal

# ╔═╡ 3c2865b6-86be-4837-957c-9e563325b652
using BenchmarkTools: @benchmark

# ╔═╡ d0d1cb0b-ed53-4c5c-9053-5d135917b34c
using Profile

# ╔═╡ dd0028ac-58fd-4ffc-afa4-57c25f47686e
using ProfileSVG: @profview

# ╔═╡ eecf828b-9756-43cb-9c27-03a55279b2b2
using JET: @report_opt

# ╔═╡ 55ff3336-0fe8-4f26-a1f8-dea3ff7f5473
md"""
> I'm lately doing for the first time some optimizations of Julia code and I sort of find it super beautiful.

This is how I started a message on the Julia language Slack in response to a question about why optimising Julia code is so difficult compared to other languages.
In the message I argued against that claim.
Optimising isn't hard in Julia if you compare it to Python or R where you have to be an expert in Python or R **and** C/C++.
Also, in that message I went through a high-level overview of how I approached optimising.
The next day, Frames Catherine White, who is a true Julia veteran, suggested that I write a blog post about my overview, so here we are.

In this blog post, I'll describe what _type stability_ is and why it is important for performance.
Unlike most other posts, I'll discuss it in the context of performance (raw throughput) and in the context of time to first X (TTFX).
Julia is sort of notorious for having really bad TTFX in certain cases.
For example, creating a plot with the [Makie.jl](https://github.com/JuliaPlots/Makie.jl) package takes 40 seconds at the time of writing.
On the second call, it takes about 0.001 seconds.
This blog post explains the workflow that you can use to reduce running time and TTFX.

## Type stability

Let's first talk about that _type stability_ thing that everyone keeps talking about.
Why is it important?
To show this, let's write naive Julia code.
Specifically, for this example, we write code which can hide the type from the compiler, that is, we need to add some kind of indirection so that the compiler cannot infer the types.
This can be done via a dictionary.
Note that our dictionary returns different types, namely an `Float32` and a `Float64`:
"""

# ╔═╡ 9dbfb7d5-7035-4ea2-a6c0-efa00e39e90f
numbers = Dict(:one => 1f0, :two => 2.0);

# ╔═╡ a36ef63b-436d-45e2-8edf-625df1575e7a
function double(mapping, key::Symbol)
	return 2 * mapping[key]
end;

# ╔═╡ 8b3cd5d7-a0e4-4fdf-aad6-698c3a0b3184
md"This code works, we can pass `:one` or `:two` and the number will be doubled:"

# ╔═╡ 87ed567e-9da6-4d0b-99af-85d513344dea
double(numbers, :one)

# ╔═╡ 3bbab8a4-62b3-4b4b-8559-2efe5fcc4af8
double(numbers, :two)

# ╔═╡ 5839326e-34d6-4713-89b9-feee36bbc824
md"""
Let's look at the optimized LLVM code via `@code_warntype`.
Here, you can ignore the `with_terminal`; it's only needed because this blog post is running in a [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebook.
"""

# ╔═╡ 2eb57dd7-b52c-4246-aba2-24646a81151b
with_terminal() do
	@code_warntype double(numbers, :one)
end

# ╔═╡ a9012b54-7c12-4239-8716-190b06bb50d6
md"""
Ouch.
The optimized code looks quite good with one `Base.getindex` and a `2 * %1`, but we do get some big red warnings about the output type which is an `Any`.
That color indicates that something is wrong.
What is wrong is that an `Any` type cannot easily be put into a memory spot.
For a concrete type such as `Float64`, we know how much space we need so we don't need a pointer and we can even put the number nearer to the CPU so that it can quickly be accessed.
To see whether a type is concrete, we can use `isconcretetype`:
"""

# ╔═╡ 9d64de5f-d9b0-4e43-9831-e0910a53bdd8
isconcretetype(Float64)

# ╔═╡ beec087a-992d-46a5-9934-b2e20f462588
isconcretetype(AbstractFloat)

# ╔═╡ 3ca857f7-6260-4e15-930a-86de5331393d
md"""
To make matters worse, Julia does a lot of optimizing, but it cannot do much for abstract types.
For example, let's write two very simple functions:
"""

# ╔═╡ da60c86b-ed10-4475-988c-099d5929946e
function use_double(mapping, x)
	doubled = 2 * double(mapping, x)
	string(doubled)
end;

# ╔═╡ 4bd3fa90-008e-4d88-a274-f6f1a0af7634
use_double(numbers, :one)

# ╔═╡ f9c00b9f-f196-426a-a7ca-9f7a83ba94ad
md"This is how the `@code_warntype` looks:"

# ╔═╡ 113167e6-baaf-4bd1-8834-745140be34c0
with_terminal() do
	@code_warntype use_double(numbers, :one)
end

# ╔═╡ a2ff5e5c-278f-41fb-8be4-ea40242886ca
md"""
The `Any` type propagated.
Now, also the `use_naive_double` function has an `Any` output type.
**And**, the type of the variable `doubled` isn't known when the function is compiled meaning that the call `string(doubled)` ends up being a _runtime dispatch_.
This means that Julia has to lookup the right method during running time in the method lookup table.
If the type was known, Julia would just hardcode the link to the right method and thus avoid a method table lookup or it would just copy-paste the content of the function to avoid jumping at all.
This is called inlining.

To see that in action, let's go on a little digression and take a look at optimised code for the case when the types **are** known.
For this, consider two simple functions:
"""

# ╔═╡ 0e9b74c1-723d-4ca1-8526-c77c12a17ee0
inner(x) = 2 * x;

# ╔═╡ 879d0d78-0c25-45d6-92be-9350b85def96
outer(x) = 3 * inner(x);

# ╔═╡ c842aaa0-92bf-42be-a970-42e6c12cde78
md"We can now call this for, say an `Int` and get an output:"

# ╔═╡ 9232bca5-a3b3-4050-abfe-a30e3f306386
outer(2)

# ╔═╡ 7b1776ed-394e-4ca8-bc2b-99382532c598
md"Let's look at the LLVM code for this function:"

# ╔═╡ cb4746fc-69d2-4f9b-a417-421392cc5ad1
with_terminal() do
	@code_llvm outer(2)
end

# ╔═╡ 554c9da7-dc70-4182-a673-8fa75d37aabe
md"""
Hopefully, you're now thinking "WOW!".
The compiler figured out that `inner` is just `2 * x` so there is no need to step into that function, we can just calculate `2 * x` directly.
But then, it figures out that `2 * 3 * x = 6 * x`, so we can get the answer in **one** LLVM instruction.

On the other hand, what if we add a `Base.inferencebarrier` to block inference inside the outer function:
"""

# ╔═╡ e02d6f85-22da-4262-9dad-3911c2222280
blocked_outer(x) = 3 * inner(Base.inferencebarrier(x));

# ╔═╡ 0ef3a4c4-96a6-409d-8686-12bec445ea1b
with_terminal() do
	@code_llvm blocked_outer(2)
end

# ╔═╡ 0e8bb623-c7b7-40aa-b02d-5b6031ad86a7
md"""
To see the difference in running time, we can compare the output `@benchmark` for both:
"""

# ╔═╡ cfac8d80-c484-4ac3-be73-9e6210873320
@benchmark outer(2)

# ╔═╡ 79c8da52-bc9c-48ab-8a52-a4ed916113a6
@benchmark blocked_outer(2)

# ╔═╡ 4e92f082-3d06-4b04-8a33-043f2a07903d
md"""
So, even though benchmarks below 1 ns aren't reliable, we can see that the inferable function (`outer`) is much faster.
Next, we'll show that this is not all due to having the extra call to `Base.inferencebarrier`.
"""

# ╔═╡ 6206b0b2-348b-44ed-9c05-6b967210fb54
md"""
We've seen that knowing the types is important for the compiler, so let's improve the type inference for the function above.
We could fix it in a few ways.
We could add a type hint at the function.
For example, a type hint could look like this:
"""

# ╔═╡ f6bece65-9735-4e4b-9e74-99735013fb93
function with_type_hint(x)
	Base.inferrencebarrier(x)::Int
end;

# ╔═╡ 3a43b076-cc9e-4dda-9dcc-d7fbb57cce15
md"With this, the output type of the function body is known:"

# ╔═╡ eb03fcfd-ec9e-422d-b2aa-de564b8f028b
with_terminal() do
	@code_warntype with_type_hint(1)
end

# ╔═╡ 0e80aee4-0cc9-43da-b081-85d3f6d1013a
md"""
which solves further inference problems if we use this method, but it is a bit risky.
The `Core.typeassert` will assert the type and throw an error if the type turns out to be wrong.
This hinders writing generic code.
Also, it takes the system a little bit of time to actually assert the type.

So, instead it would be better to go to the root of the problem.
Above, we had a dictionary `numbers`:
"""

# ╔═╡ 2415cdc1-a178-46b4-aaee-735d233af6c1
numbers

# ╔═╡ 530f5310-4ac0-4754-8574-5eb0b408534d
md"The type is:"

# ╔═╡ 8a2c2500-cfeb-42e4-a29a-d06f544900f0
typeof(numbers)

# ╔═╡ be708e35-ced8-44ac-a0e7-fbdc7cfe1a33
md"""
Where `AbstractFloat` is a non-concrete type meaning that it cannot have direct instance values, and more importantly meaning **that we cannot say with certainty which method should be called for an object of such a type**.
"""

# ╔═╡ d1c9f0f7-010e-4be8-ae6e-dbd6dd791d6c
md"
We can make this type concrete by manually specifying the type of the dictionary.
Now, Julia will automatically convert our `Float32` to a `Float64`:
"

# ╔═╡ d95c0dd4-e6c4-4328-9e50-44104e426099
typednumbers = Dict{Symbol, Float64}(:one => 1f0, :two => 2.0);

# ╔═╡ 73120fa2-ed0a-4170-a9fe-cccc42e4c6ec
md"Let's look again to the `@code_warntype`:"

# ╔═╡ 00e9b1a5-c2cc-4e57-a9fb-edaba6696684
with_terminal() do
	@code_warntype use_double(typednumbers, :one)
end

# ╔═╡ ba536164-78a2-4080-acb4-d6d3aeaa1e38
md"""
Great!
So, this is now exactly the same function as above, but all the types are known and the compiler is happy.

Let's run the benchmarks for both `numbers` and `typednumbers`:
"""

# ╔═╡ 20188300-6dd6-465a-9be0-be91fc540674
@benchmark use_double(numbers, :one)

# ╔═╡ 91f2b4b0-0276-453a-9084-43ac497df0d4
@benchmark use_double(typednumbers, :one)

# ╔═╡ cb0a2558-e55d-406a-8394-bf1a96a6e38d
md"""
So, that's a reduction in running time which we basically got for free.
The only thing we needed to do was look through our naive code and help out the compiler a bit by adding more information.

And this is exactly what I find so beautiful about the Julia language.
You have this high-level language where you can be very expressive, write in whatever style you want and don't have to bother about putting type annotations on all your functions.
Instead, you first focus on your proof of concept and get your code working and only **then** you start digging into optimizing your code.
To do this, you can often get pretty far already by looking at `@code_warntype`.

But, what if your code contains more than a few functions?
Let's take a look at some of the available tooling.
"""

# ╔═╡ 68e2b2c1-8cb6-4ef9-88f8-099a00037a72
md"""
## Tooling

The most common tool for improving performance is a profiler.
Julia has a profiler in the standard library:
"""

# ╔═╡ c18352f7-6c64-4e3d-bc1e-6f16e0b6cea0
md"This is a sampling-based profiler meaning that it takes samples to estimate how much time is spent in each function."

# ╔═╡ b9bb12ad-6b66-402c-8cde-4d0e2f266d27
@profile foreach(x -> blocked_outer(2), 1:100)

# ╔═╡ 54f83adf-e8cd-4d16-bfa9-2413638bf6e9
md"""
We can now call `Profile.print()` to see the output and how many samples were taken in each function.
However, in most cases we want to have a nice plot.
Here, I use [ProfileSVG.jl](https://github.com/kimikage/ProfileSVG.jl), but other options are also listed in the [Julia Profiling documentation](https://docs.julialang.org/en/v1/manual/profile/).
See especially [PProf.jl](https://github.com/JuliaPerf/PProf.jl) since that viewer can show graphs as well as flame graphs.
"""

# ╔═╡ a3cbe176-d1b9-485c-b655-ee9f164e84a3
@profview foreach(x -> blocked_outer(2), 1:10_000_000)

# ╔═╡ 97de2cf4-fd71-4ab9-b892-fe36aced2c57
md"""
In this image, you can click on an element to see the location of the called function.
Unfortunately, because the page that you're looking at was running inside a Pluto notebook, this output shows a bit of noise.
You can focus on everything above `eval in boot.jl` to see where the time was spent.
In essence, the idea is here that the wider a block, the more time is spent on it.
Also, blocks which lay on top of other block indicate that they were called inside the outer block.
As can be seen, the profiler is very useful to get an idea of which function takes the most time to run.

However, this doesn't tell us **what** is happening exactly.
For that, we need to dive deeper and look critically at the source code of the function which takes long.
Sometimes, that already provides enough information to see what can be optimized.
In other cases, the problem isn't so obvious.
Probably, there is a type inference problem because that can make huge differences as is shown in the section above.
One way would then be to go to the function which takes the most time to run and see how the type inference looks via `@code_warntype`.
Unfortunately, this can be a bit tricky.
Consider, for example, a function with keyword arguments:
"""

# ╔═╡ 8a64d202-811c-4189-b306-103054acdc28
with_keyword_arguments(a; b=3) = a + b;

# ╔═╡ bbc1187b-764c-4b57-ac6b-90a10350f234
with_terminal() do
	@code_warntype with_keyword_arguments(1)
end

# ╔═╡ b52be1dd-56d1-43d3-8795-f10c195d13bd
md"""
Here, we don't see the `a + b` as we would expect, but instead see that the `with_keyword_arguments` calls another function without keyword arguments.
Now, we would need to manually call this nested function with a generated name `var"#with_keyword_arguments#1"` with exactly the right inputs to see what `@code_warntype` does exactly inside this function.
Even worse, imagine that you have a function which calls a function which calls a function...

To solve this, there is [Cthulhu.jl](https://github.com/JuliaDebug/Cthulhu.jl).
With Cthulhu, it is possible to `@descend` into a function and see the code warntype.
Next, the arrow keys and enter can be used to step into a function and see the code warntype for that.
By continuously stepping into and out of functions, it is much easier to see what code is calling what and where exactly the type inference starts to fail.
Often, by solving a type inference problem at exactly the right spot, inference problems for a whole bunch of functions can be fixed.
For more information about Cthulhu, see the GitHub page linked above.
"""

# ╔═╡ bc0fc838-0f93-42de-81f4-a1a8c7dabfa0
md"""
A complementary tool to find the root of type problems is [JET.jl](https://github.com/aviatesk/JET.jl).
Basically, this tool can automate the process described above.
It relies on Julia's compiler and can point to the root of type inference problems.
Let's do a demo.
Here, we use the optimization analysis:
"""

# ╔═╡ 4a359e9e-e71a-4beb-bb94-408535182868
@report_opt blocked_outer(2)

# ╔═╡ 05751ac5-2eae-4bab-9ecc-50949523d02a
md"""
In this case, the tool points out exactly the problem we've had.
Namely, because the function definition is `3 * inner(Base.inferencebarrier(x))`, the `inner` function call cannot be optimized because the type is unknown at that point.
Also, the output of `inner(Base.inferencebarrier(x))` is unkown and we have another runtime dispatch.

For extremely long outputs, it can be useful to print the output of JET to a file to easily navigate through the output.

These are the most important tools to improve performance.
If this is all you care about, then feel free to stop reading here.
In the next section, let's take a look at how to reduce the time to first X.
"""

# ╔═╡ 70f4193a-5ad6-496f-9107-ee54c3d6da33
md"""
## Precompilation

As described above, Julia does lots of optimizations on your code.
For example, it removes unnecessary function calls and hardcodes method calls if possible.
This takes time and that is a problem.
Like said above, Makie runs extremely quick after the first time that you have created a plot going from 40 seconds to something like 0.001 seconds.
And, we need to wait all these seconds every time that we restart Julia.
Of course, Julia developers don't develop by changing their plotting code and wait 40 seconds to see the output.
We use tools such as [Pluto.jl](https://github.com/fonsp/Pluto.jl) or [Revise.jl](https://github.com/timholy/Revise.jl) to use code changes without restarting Julia.
Still, sometimes it is necessary to restart Julia, so what can we do to reduce the compilation time?

Well, we can reduce the compilation time by shouting **I am the compiler now!** and write optimized code manually.
For example, this is done in [OrdinaryDiffEq.jl#1465](https://github.com/SciML/OrdinaryDiffEq.jl/pull/1465).
In some cases, this can be a great last-resort solution to make some compilation time disappear.

However, it is quite laborious and not suitable in all cases.
A very nice alternative idea is to move the compilation time into the _precompilation_ stage.
Precompilation occurs right after package installation or when loading a package after it has been changed.
The results of this compilation are retained even after restarting the Julia instance.
So, instead of having to compile things for each restart, we just compile it only when changing the package!
Sounds like a good deal.

It is a good deal.
Except, we have to note that we're working with the Julia language.
Not all functions have typed arguments let alone concretely typed arguments, so the precompile phase cannot always know **what** it should compile.
Even more, Julia by default doesn't compile all functions with concretely typed arguments.
It just assumes that some function will probably not be used, so no need to precompile it.
This is on purpose, to avoid developers putting concrete types everywhere which would make Julia packages less composable which is a very fair argument.

Anyway, we can fix this by adding precompile directives ourselves.
For example, we can create a new function, call `precompile` on it for integers and look at the existing method specializations:
"""

# ╔═╡ 7cdb3017-e391-4558-ba13-3eed113c99b1
begin
	add_one(x) = x + 1
	precompile(add_one, (Int,))
	methods(add_one)[1].specializations
end

# ╔═╡ 43af319b-af41-4b3a-87bd-3cc1f6069d9b
md"""
A method specialization is just another way of saying a compiled instance for a method.
So, a specialization is always for some concrete types.
This method specialization shows that `add_one` is compiled even though we haven't called `add_one` yet.
The function is completely ready for use for integers.
If we pass another type, the function would still need to compile.

What is nice about this is that the `precompile` will compile everything recursively.
So, say, we have a large codebase handling some kind of notebooks and the package has some kind of `open` function with concrete types such as a `ServerSession` to open the notebook into and a `String` with the path for the notebook location, then we can add a precompile on that function as follows:

```julia
precompile(open, (ServerSession, String))
```

Inside this large codebase.
Since the `open` function is calling many other functions, the `precompile` will compile many functions and can reduce the time to first X by a lot.
This is what happened in [Pluto.jl#1934](https://github.com/fonsp/Pluto.jl/pull/1934).
We've added **one line of code** to reduce the time to first open a notebook from 11 to 8 seconds.
That is a 30% reduction in running time by adding one line of code.
To figure out where you need to add precompile directives exactly, you can use [SnoopCompile.jl](https://github.com/timholy/SnoopCompile.jl).

Alas, now you probably wonder why we didn't have a 100% reduction.
The answer is type inference.
`precompile` will go through all the functions recursively but once the type becomes non-concrete, it cannot know what to compile.
To fix this, we can use the tools presented above to fix type inference problems.

In conclusion, this is what I find so beautiful about the language.
You can hack your proof-of-concept together in very naive ways and then throw on a few precompiles if you want to reduce the TTFX.
Then, once you need performance, you can pinpoint what method takes the most time, look at the generated LLVM code and start fixing problems such as type inference.
Improving the inferability will often make code more readable, it will reduce running time **and** it will reduce time to first X; all at the same time.
"""

# ╔═╡ fa412ecb-959a-417d-a508-3d428debf801
md"""
## Acknowledgements

Thanks to [Michael Helton](https://github.com/heltonmc), [Rafael Fourquet](https://github.com/rfourquet) and [Guillaume Dalle](https://gdalle.github.io/) for providing feedback on this blog post.
"""

# ╔═╡ 3abf883e-064d-486d-9a73-b03719679255
md"""
## Appendix
"""


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
JET = "c3a54625-cd67-489e-a8e7-0a5a0ff4e31b"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
ProfileSVG = "132c30aa-f267-4189-9183-c8a63c7e05e6"

[compat]
BenchmarkTools = "~1.5.0"
JET = "~0.9.12"
PlutoUI = "~0.7.60"
ProfileSVG = "~0.2.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "c5533ccf06873d3a176de6dca350e6c02e9eaa44"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.FlameGraphs]]
deps = ["AbstractTrees", "Colors", "FileIO", "FixedPointNumbers", "IndirectArrays", "LeftChildRightSiblingTrees", "Profile"]
git-tree-sha1 = "c8bb515422866a684d9e67870fc5791e3292ad01"
uuid = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"
version = "1.0.1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JET]]
deps = ["CodeTracking", "InteractiveUtils", "JuliaInterpreter", "LoweredCodeUtils", "MacroTools", "Pkg", "PrecompileTools", "Preferences", "Test"]
git-tree-sha1 = "5c5ac91e775b585864015c5c1703cee283071a47"
uuid = "c3a54625-cd67-489e-a8e7-0a5a0ff4e31b"
version = "0.9.12"

    [deps.JET.extensions]
    JETCthulhuExt = "Cthulhu"
    ReviseExt = "Revise"

    [deps.JET.weakdeps]
    Cthulhu = "f68482b8-f384-11e8-15f7-abe071a5a75f"
    Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "10da5154188682e5c0726823c2b5125957ec3778"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.38"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "688d6d9e098109051ae33d126fcfc88c4ce4a021"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProfileSVG]]
deps = ["Colors", "FlameGraphs", "Profile", "UUIDs"]
git-tree-sha1 = "95ef58783baaa61cd227342b7b605d8a4dfba4a9"
uuid = "132c30aa-f267-4189-9183-c8a63c7e05e6"
version = "0.2.2"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═55ff3336-0fe8-4f26-a1f8-dea3ff7f5473
# ╠═9dbfb7d5-7035-4ea2-a6c0-efa00e39e90f
# ╠═a36ef63b-436d-45e2-8edf-625df1575e7a
# ╠═8b3cd5d7-a0e4-4fdf-aad6-698c3a0b3184
# ╠═87ed567e-9da6-4d0b-99af-85d513344dea
# ╠═3bbab8a4-62b3-4b4b-8559-2efe5fcc4af8
# ╠═5839326e-34d6-4713-89b9-feee36bbc824
# ╠═87efa960-4dbb-42c3-9ad5-7189f58beeb5
# ╠═2eb57dd7-b52c-4246-aba2-24646a81151b
# ╠═a9012b54-7c12-4239-8716-190b06bb50d6
# ╠═9d64de5f-d9b0-4e43-9831-e0910a53bdd8
# ╠═beec087a-992d-46a5-9934-b2e20f462588
# ╠═3ca857f7-6260-4e15-930a-86de5331393d
# ╠═da60c86b-ed10-4475-988c-099d5929946e
# ╠═4bd3fa90-008e-4d88-a274-f6f1a0af7634
# ╠═f9c00b9f-f196-426a-a7ca-9f7a83ba94ad
# ╠═113167e6-baaf-4bd1-8834-745140be34c0
# ╠═a2ff5e5c-278f-41fb-8be4-ea40242886ca
# ╠═0e9b74c1-723d-4ca1-8526-c77c12a17ee0
# ╠═879d0d78-0c25-45d6-92be-9350b85def96
# ╠═c842aaa0-92bf-42be-a970-42e6c12cde78
# ╠═9232bca5-a3b3-4050-abfe-a30e3f306386
# ╠═7b1776ed-394e-4ca8-bc2b-99382532c598
# ╠═cb4746fc-69d2-4f9b-a417-421392cc5ad1
# ╠═554c9da7-dc70-4182-a673-8fa75d37aabe
# ╠═e02d6f85-22da-4262-9dad-3911c2222280
# ╠═0ef3a4c4-96a6-409d-8686-12bec445ea1b
# ╠═0e8bb623-c7b7-40aa-b02d-5b6031ad86a7
# ╠═3c2865b6-86be-4837-957c-9e563325b652
# ╠═cfac8d80-c484-4ac3-be73-9e6210873320
# ╠═79c8da52-bc9c-48ab-8a52-a4ed916113a6
# ╠═4e92f082-3d06-4b04-8a33-043f2a07903d
# ╠═6206b0b2-348b-44ed-9c05-6b967210fb54
# ╠═f6bece65-9735-4e4b-9e74-99735013fb93
# ╠═3a43b076-cc9e-4dda-9dcc-d7fbb57cce15
# ╠═eb03fcfd-ec9e-422d-b2aa-de564b8f028b
# ╠═0e80aee4-0cc9-43da-b081-85d3f6d1013a
# ╠═2415cdc1-a178-46b4-aaee-735d233af6c1
# ╠═530f5310-4ac0-4754-8574-5eb0b408534d
# ╠═8a2c2500-cfeb-42e4-a29a-d06f544900f0
# ╠═be708e35-ced8-44ac-a0e7-fbdc7cfe1a33
# ╠═d1c9f0f7-010e-4be8-ae6e-dbd6dd791d6c
# ╠═d95c0dd4-e6c4-4328-9e50-44104e426099
# ╠═73120fa2-ed0a-4170-a9fe-cccc42e4c6ec
# ╠═00e9b1a5-c2cc-4e57-a9fb-edaba6696684
# ╠═ba536164-78a2-4080-acb4-d6d3aeaa1e38
# ╠═20188300-6dd6-465a-9be0-be91fc540674
# ╠═91f2b4b0-0276-453a-9084-43ac497df0d4
# ╠═cb0a2558-e55d-406a-8394-bf1a96a6e38d
# ╠═68e2b2c1-8cb6-4ef9-88f8-099a00037a72
# ╠═d0d1cb0b-ed53-4c5c-9053-5d135917b34c
# ╠═c18352f7-6c64-4e3d-bc1e-6f16e0b6cea0
# ╠═b9bb12ad-6b66-402c-8cde-4d0e2f266d27
# ╠═54f83adf-e8cd-4d16-bfa9-2413638bf6e9
# ╠═dd0028ac-58fd-4ffc-afa4-57c25f47686e
# ╠═a3cbe176-d1b9-485c-b655-ee9f164e84a3
# ╠═97de2cf4-fd71-4ab9-b892-fe36aced2c57
# ╠═8a64d202-811c-4189-b306-103054acdc28
# ╠═bbc1187b-764c-4b57-ac6b-90a10350f234
# ╠═b52be1dd-56d1-43d3-8795-f10c195d13bd
# ╠═bc0fc838-0f93-42de-81f4-a1a8c7dabfa0
# ╠═eecf828b-9756-43cb-9c27-03a55279b2b2
# ╠═4a359e9e-e71a-4beb-bb94-408535182868
# ╠═05751ac5-2eae-4bab-9ecc-50949523d02a
# ╠═70f4193a-5ad6-496f-9107-ee54c3d6da33
# ╠═7cdb3017-e391-4558-ba13-3eed113c99b1
# ╠═43af319b-af41-4b3a-87bd-3cc1f6069d9b
# ╠═fa412ecb-959a-417d-a508-3d428debf801
# ╠═3abf883e-064d-486d-9a73-b03719679255
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
