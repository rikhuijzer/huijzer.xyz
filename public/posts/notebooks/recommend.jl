### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ d1631598-5682-421c-988f-73b137dd6b74
using Measurements: measurement

# ╔═╡ 6b727e5f-605e-405b-b89b-863275a165a6
md"""
Yuri Vishnevsky [wrote](https://yuri.is/not-julia/) that he no longer recommends Julia.
This caused lengthy discussions at [Hacker News](https://news.ycombinator.com/item?id=31396861), [Reddit](https://www.reddit.com/r/programming/comments/uqznl3/the_julia_language_has_a_number_of_correctness/) and the [Julia forum](https://discourse.julialang.org/t/discussion-on-why-i-no-longer-recommend-julia-by-yuri-vishnevsky/81151).
Yuri argues that Julia shouldn't be used in any context where correctness matters.
Based on the amount and the ferocity of the comments, it is natural to conclude that Julia as a whole must produce incorrect results and therefore cannot be a productive environment.
However, the scope of the blog post and the discussions are narrow.
In general, I still recommend Julia for data science applications because it is fundamentally productive and, with care, correct.
"""

# ╔═╡ c16d8f63-e495-4ac2-b1b4-e7d38c9c1156
md"""
## Classless

To understand Julia, we have to first understand Julia's lack of classes.
In object-oriented programming, classes allow objects and methods on those objects to be put in one space.
The idea is that when you are, for example, writing business logic with books, then you want to neatly put the operations on the book near the definition of the book.
One such operation can be to `sell`.
The `sell` function can do things like marking the book object as sold.
This can work fine, you can even add new classes such as bicycles and spoons and add a `sell` method to them as long as they all inherit from the same superclass, say, `Product`.
This `Product` class would define an empty `sell` method which needs to be extended by all subclasses and allows the compiler and static linter tools to know that any `Product` subclass has a `sell` method which is great for discoverability of those methods.
So, as an object-oriented programmer, a lot of thought is put into finding the right inheritance so that methods can be re-used between different classes.
Also, class and object variables hold the data and are coupled to methods too.
In summary, methods are coupled to other methods and to data.

This coupling is counter to the idea of modularity.
Modular artifacts are easier to produce and maintain.
For example, mechanical artifacts such as cars, bridges and bikes are not glued together but instead use fasteners such as nuts and bolts.
If the products would be all glued together, then the tiniest mistake in the production of one of these artifacts means that it can be thrown away immediately.
The same holds for maintenance.
In most cases, repairing a glued together artifact is more expensive than replacing it by a new artifact.
And this is exactly the problem with object-oriented programming.
In essence, the whole system is glued together since not only methods are glued to their class and data, but also classes are glued together.

This leads us to the description of Julia's class-system.
Julia doesn't have a class-system.
Instead of linking functions to classes, all functions are separate objects which can be located via the global method table.
Before we can dive into that, we have to start by describing how function calls work in Julia.
"""

# ╔═╡ 402babbc-4027-4e7a-81f8-44dc7c63a466
md"""
## Multiple dispatch

Instead of a hierarchy on objects, Julia implements a hierarchy on types.
For example, the code
"""

# ╔═╡ b7dba04e-8158-4251-bfb5-3f0143356ce1
plus_one(x::Number) = x + 1;

# ╔═╡ 7a5b5fc6-ab2c-4a43-843d-76630c1a94dd
md"""
defines a function `plus_one` with one method.
In this blog post, the semicolon denotes that the output shouldn't be printed.
The method takes one object `x` of type `Number`.
We can extend this function by adding another method:
"""

# ╔═╡ 8168d317-65cd-41e7-97b8-3ae90b304649
plus_one(x::String) = "$(x) 1";

# ╔═╡ 346be37e-8145-4f17-a061-fe84d871bfcb
md"""
Now, calling the function with a number will increase the number by one and calling it with a string will add a 1 at the end:
"""

# ╔═╡ 91739577-2440-4dc5-8fc6-b693ab006917
md"""
Important to note is that Julia will always call the most specialized methods for any type.
For instance, we could add another method for `Float64`:
"""

# ╔═╡ d8ef4dca-b9ac-458d-bd11-2b8eb2b2c84c
plus_one(x::Float64) = error("Float64 is not supported");

# ╔═╡ e7bcb1a6-7846-4663-b25f-dd646707e8e2
md"""
to cause the function call to error for an 64-bit floating point number.
We can also add a fallback with
"""

# ╔═╡ 340843db-f27f-4b6b-8f08-b018a61fa59c
plus_one(x) = error("Type $(typeof(x)) is not supported");

# ╔═╡ cf58cbab-5fda-4f82-8ce0-4c7434b7e9d1
plus_one(0)

# ╔═╡ 22234612-d9d3-4a77-a36c-9591cb2739b5
plus_one("my text")

# ╔═╡ 05895471-2c69-4d87-b09c-a0e1f5677afc
md"""
to get
"""

# ╔═╡ 67ee2625-c7cc-4937-b153-d934329f6eb5
try # hide
plus_one(Symbol("my text"))
catch e # hide
    e # hide
end # hide

# ╔═╡ 65cc687c-b764-465b-adbe-8c286ab6035d
md"""
Each addition of a method modifies the method tables.
For each function, there is a table which allows Julia's compiler to convert each function call to the right method call.
The table can also handle dispatch on multiple types, which is much more involved, hence the name multiple dispatch.
This is what enables Julia's composability.
"""

# ╔═╡ 2361a1cf-f1f4-401a-8a6b-ed83f9b1c14d
md"""
## Composability

Due to these method tables, Julia allows arbitrary combinations of packages to work together.
For example, the [`Measurements.jl`](https://github.com/JuliaPhysics/Measurements.jl) package can be used to propagate uncertainties caused by physical measurements.
We can define such a measurement as follows:
"""

# ╔═╡ b1568a49-a7bc-4f2a-90fc-887e00d1dac5
u = measurement(4.5, 0.1)

# ╔═╡ 76bff2f8-4ba3-463e-8cb2-6b3cca626b82
md"""
and pass the measurement into `plus_one`:
"""

# ╔═╡ f3f64230-02ad-4264-9a7e-6db9c33d7608
plus_one(u)

# ╔═╡ 24e4e976-e322-4d36-9b75-cbc331d88787
md"""
Even though `plus_one` was written without having `Measurements` in mind, it still works.
The reason that it works is that `plus_one` calls the function `+` on its input argument `x` and `1`.
Julia will look this up in the method table for `+` and realize that it has to call a `+` method defined by `Measurements` which handles the operation.
And this is what surprises me a bit in the aforementioned critique of Julia.
The critique was that many packages don't always produce the right results when used together.
Basically, the critique is taking the standpoint that there is a huge combination of packages that can be misused together and so the language is inherently broken.
Instead, I'd say that there is a huge combination of packages that can be used together providing the language with an enormous amount of possibilities.
It is up to the programmer to verify the interaction before use and, preferably, add tests to one of the packages.
In other languages, the programmer would need to manually write glue code to make packages work together.
Like said above, glued together code is hard to create and maintain.
"""

# ╔═╡ fc140bde-6b9e-4932-9ca1-51afd46a24f6
md"""
## Performance

As mentioned before, much happens via the method tables.
However, such a lookup isn't quick.
Imagine having a loop over thousands of numbers and doing a table lookup in each iteration.
Julia's trick is to solve this during the compilation phase and to generate highly-optimized LLVM code.
This allows Julia programs to be quick if desired.
In a nutshell, if you want to optimize Julia code, the goals are to reduce allocations, improve type stability and improve inferability.
These targets can be achieved by many introspecting tools provided by the language such as `@code_warntype` and `@code_llvm` which show, respectively, the generated typed and LLVM code.
For more information, see my blog post on [optimizing Julia code](https://huijzer.xyz/posts/inference/).
Therefore, in comparison to many other high-level languages, Julia is one of the few where writing fast code doesn't require expertise in low and high-level languages at the same time, and one of the few where you don't have to glue your high-level code to low-level code.
"""

# ╔═╡ 98c2ff11-af50-4531-af99-9a184e122b40
md"""
## Compilation time

A valid critique of Julia is the compilation time or, actually, the time it takes to get to first X (TTFX) where X can be anything such as a HTTP response, a plot or prediction from a model.
Julia's dynamic nature, the expressiveness that the compiler allows, and the decision to aggressively optimize the LLVM code causes much time to be spent in compilation.
In practise, loading a few libraries and calling a few functions can easily take a few minutes whereas the second call to the same function can take less than a second.
Restarting the Julia process will trigger compilation from scratch again.
Mitigations are to keep the process alive and live reload code.
Tools for this are [Revise.jl](https://github.com/timholy/Revise.jl) or [Pluto.jl](https://github.com/fonsp/Pluto.jl).
With this, Julia's compilation time is faster than ahead of time compiled languages because only the methods that changed will be recompiled.
Hence, the development workflow is to start the process, get a cup of coffee while it loads for the first time and keep the process alive for the rest of the day.

As a side note, there is hope that the compilation time will be reduced further.
There are various complementary angles of attack.
For example, Julia 1.8 reduces compilation time [up to 50%](https://github.com/JuliaLang/julia/pull/43990#issuecomment-1044612013) and this could be reduced further by [caching binary code in Julia 1.9](https://github.com/JuliaLang/julia/pull/44527).
In many cases, the compilation time can also be reduced by writing easy-to-compile code which often is synonymous for easy-to-read code.
Finally, it is also likely that CPUs will become faster over time and that tooling around reducing TTFX will improve.
"""

# ╔═╡ 64b2d071-b0b7-4f53-976f-92f0aa50a72a
md"""
## Consistency

One factor influencing how easy it is to learn a language is consistency.
The more consistent a language is, the less rote memorization is needed for edge cases.
This is something that has to be done right from the start of a language since breaking changes are hard to sell to users.
Julia has spent much effort in getting the APIs right in Julia 1.0.
One such example is code vectorization.
In many languages, writing fast code for arrays means passing the array into function as a whole.
Only via that way, the function can apply optimizations such as parallelization or [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data).
However, when writing a function that takes an array what should it return?
Logical reasoning would say an array.
Some language communities say an array, a dataframe, a vector of strings containing text and the numbers or, well, anything.
As a user, you're required to learn these idiosyncrasies by heart.
In Julia, the idea is to instead let the user decide how to call functions at the caller side and optimize that.
For example, users can easily write SIMD loops via [`LoopVectorization.jl`](https://github.com/JuliaSIMD/LoopVectorization.jl) or use the built-in broadcasting which generates specialized code for the loops.
For example, we can apply broadcasting via the dot operator on the `plus_one` function:
"""

# ╔═╡ 40247787-626a-4e93-8a52-94df677c4917
plus_one.([1, 2, 3])

# ╔═╡ 35f945e9-41b3-4905-8db4-7f2a09dcbd62
md"""
which will be optimized to a SIMD loop as can be verified in the REPL via `@code_typed plus_one.([1, 2, 3])` or `@code_llvm plus_one.([1, 2, 3])`.
"""

# ╔═╡ 3550b61a-8a48-48fc-a88a-51a27ff40f1c
md"""
## Operating system independence

Another reason that I still recommend Julia is that it manages platform independence well.
With this, I mean that packages as a whole are easy to move between operating systems; not just the language itself.
To do this, Julia has solved problems with package management and third-party binary objects.

Package management is handled by the built-in package manager [`Pkg.jl`](https://github.com/JuliaLang/Pkg.jl).
Simply put, instead of spending efforts on multiple package managers, one is maintained by the core language team and used by everyone.
The manager supports environments, like Python's `virtualenv`, and a manifest specifying all dependencies and indirect dependencies for reproducibility.
These environments allow installing multiple versions of the same package at the same time and supports private registries in support of corporate usage.
In practice, `Pkg.jl` works flawlessly and is easy-to-use.

At the same time, third-party binaries, such as the LaTeX engine Tectonic or the GR Framework for visualization applications, can be provided via the [`Artifacts.jl`](https://pkgdocs.julialang.org/v1/artifacts/) system.
Binaries can be specified by an URL and a hash.
Then, the system will automatically download the right binaries during package installation and cache them.
On top of this, [Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil) has goes one step futher and bundles third-party binaries.
It does this via an automated build system comparable to [nixpkgs](https://github.com/nixos/nixpkgs).
The system compiles many binaries from scratch to support as many platforms as possible.
Now a crazy fact.
Often, Yggdrasil's binaries are available for more platforms than the original binaries!
Like `nixpkgs`, the binaries can be dynamically linked to other binaries already existing in the system, say, upstream binaries.
This avoids having to statically compile everything and, hence, create large binaries.
For the interested reader, it works by wrapping the environment in a temporary environment with links to the upstream binaries.

In practice, these systems mean that you almost never need to configure your operating system.
In other words, you only have to install Julia or get the default Julia Docker image.
You don't need much system configuration nor some kind of virtualization for CI or production environments.
Docker or `apt-get` are mostly restricted to cases where packages aren't yet available in Julia.
"""

# ╔═╡ 22f05e99-7cd6-4456-b5b4-95b95645901b
md"""
## Expressiveness and conclusion

To finally answer why I still recommend Julia, I have to talk about expressiveness.
Expressiveness, in my opinion, is synonymous to productivity.
Expressiveness is about how concisely and readily an idea can be expressed in a language (Farmer, [2010](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.304.6705&rep=rep1&type=pdf)).
It is similar to Dijkstra's idea that programs should be elegant.
Put in more modern terms, I'd say that a more expressive language requires fewer characters to implement an solution to a problem while maintaining human readability.

From the definition, it becomes clear that a languages expressiveness relates to the idea or problem at hand.
At this moment, I wouldn't call Julia expressive for problems where the core packages are missing, static compilation is a must-have, garbage collection is undesirable, or where low compilation times are essential.
Here, by "core packages", I mean foundational packages for things like HTTP, dataframes, images or neural networks.
However, the situation is different when the core packages are available, dynamic languages are acceptable, and garbage collection and compilation time are acceptable.
In that sweet spot, Julia shines because

- the language has shunned classes so that code is maintainable and easily extendable,
- composability allows packages and user-defined types to be combined in countless ways without much glue code,
- performant code can be written in the high-level language and doesn't need to be glued to other languages, and
- third-party binaries can be integrated in a declarative way.

When looking from this perspective, I don't see any other language coming even close to Julia in terms of expressiveness.
Even more so, as the language will reduce the time to first X and as more packages will become available, I expect its expressiveness to take the lead in many more domains.
This means that Julia is already a very productive language and that it will become a more and more productive language over time for data science applications.

That is why I still recommend Julia.

## Acknowledgements

Thanks to [Jose Storopoli](https://storopoli.io/), [Scott Paul Jones](https://github.com/ScottPJones), and [Magnus Lie Hetland](http://hetland.org/) for providing suggestions for this text.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"

[compat]
Measurements = "~2.11.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "02e77b1d3c45741082c3658bbbef332a14dae6da"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Measurements]]
deps = ["Calculus", "LinearAlgebra", "Printf", "Requires"]
git-tree-sha1 = "bdcde8ec04ca84aef5b124a17684bf3b302de00e"
uuid = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
version = "2.11.0"

    [deps.Measurements.extensions]
    MeasurementsBaseTypeExt = "BaseType"
    MeasurementsJunoExt = "Juno"
    MeasurementsRecipesBaseExt = "RecipesBase"
    MeasurementsSpecialFunctionsExt = "SpecialFunctions"
    MeasurementsUnitfulExt = "Unitful"

    [deps.Measurements.weakdeps]
    BaseType = "7fbed51b-1ef5-4d67-9085-a4a9b26f478c"
    Juno = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"
"""

# ╔═╡ Cell order:
# ╠═6b727e5f-605e-405b-b89b-863275a165a6
# ╠═c16d8f63-e495-4ac2-b1b4-e7d38c9c1156
# ╠═402babbc-4027-4e7a-81f8-44dc7c63a466
# ╠═b7dba04e-8158-4251-bfb5-3f0143356ce1
# ╠═7a5b5fc6-ab2c-4a43-843d-76630c1a94dd
# ╠═8168d317-65cd-41e7-97b8-3ae90b304649
# ╠═346be37e-8145-4f17-a061-fe84d871bfcb
# ╠═cf58cbab-5fda-4f82-8ce0-4c7434b7e9d1
# ╠═22234612-d9d3-4a77-a36c-9591cb2739b5
# ╠═91739577-2440-4dc5-8fc6-b693ab006917
# ╠═d8ef4dca-b9ac-458d-bd11-2b8eb2b2c84c
# ╠═e7bcb1a6-7846-4663-b25f-dd646707e8e2
# ╠═340843db-f27f-4b6b-8f08-b018a61fa59c
# ╠═05895471-2c69-4d87-b09c-a0e1f5677afc
# ╠═67ee2625-c7cc-4937-b153-d934329f6eb5
# ╠═65cc687c-b764-465b-adbe-8c286ab6035d
# ╠═2361a1cf-f1f4-401a-8a6b-ed83f9b1c14d
# ╠═d1631598-5682-421c-988f-73b137dd6b74
# ╠═b1568a49-a7bc-4f2a-90fc-887e00d1dac5
# ╠═76bff2f8-4ba3-463e-8cb2-6b3cca626b82
# ╠═f3f64230-02ad-4264-9a7e-6db9c33d7608
# ╠═24e4e976-e322-4d36-9b75-cbc331d88787
# ╠═fc140bde-6b9e-4932-9ca1-51afd46a24f6
# ╠═98c2ff11-af50-4531-af99-9a184e122b40
# ╠═64b2d071-b0b7-4f53-976f-92f0aa50a72a
# ╠═40247787-626a-4e93-8a52-94df677c4917
# ╠═35f945e9-41b3-4905-8db4-7f2a09dcbd62
# ╠═3550b61a-8a48-48fc-a88a-51a27ff40f1c
# ╠═22f05e99-7cd6-4456-b5b4-95b95645901b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
