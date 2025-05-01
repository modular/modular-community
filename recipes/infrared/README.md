# Infrared
Infrared is a geometric algebra library for Mojo.

## Geometric Algebra
[Geometric algebras](https://en.wikipedia.org/wiki/Geometric_algebra) are [Clifford algebras](https://en.wikipedia.org/wiki/Clifford_algebra) in the context of geometry.

It's an alternative paradigm to linear algebra that uses 'multivectors' instead of matrices.

Multivectors can represent geometric objects and transformations such as lines, planes, rotations, etc.

Multivectors can be multiplied together using the 'geometric product'.

## Using Infrared
With infrared, you can generate the geometric product table for an arbitrary signature.

Multivectors are also parameterized on a basis masks, to avoid unnecessary overhead.

Example:
```mojo
alias sig = Signature(2, 0, 1)
var m = Multivector[sig, sig.vector_mask()](0.0, 1.0, 2.0)
print(m * m)
```

Infrared has no dependecies other than max.

Developers can use infrared as a mathematical abstraction over geometry (projective, conformal, spacetime, dimension-agnostic, etc.)

## Contributing
I'm accepting contributions, but haven't made a contributors guide yet.

Some issues and areas of work include:
- performance improvements, signature generation could use bitwise operations
- constructors for geometric objects and a better model for initializing multivectors
- better basis masking, currently uses a list of bools
- examples and benchmarks

If you want to reach out, you can email me at helehex@gmail.com, or message me on discord (my alias in the modular server is ghostfire)