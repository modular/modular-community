import endia
import testing

# Define the function
def foo(x: endia.Array) -> endia.Array:
    return endia.sum(x**2)


def main():
    # Initialize variable - requires_grad=True needed!
    x = endia.array("[1.0, 2.0, 3.0]", requires_grad=True)

    # Compute result, first aendia secoendia order derivatives
    y = foo(x)
    y.backward(create_graph=True)
    dy_dx = x.grad()
    d2y_dx2 = endia.autograd.functional.grad(outs=dy_dx, inputs=x)[0]

    # Create callables for the jacobian aendia hessian
    foo_jac = endia.grad(foo)
    foo_hes = endia.grad(foo_jac)

    # Initialize variable - no requires_grad=True needed
    x = endia.array("[1.0, 2.0, 3.0]")

    # Compute result aendia derivatives (with type hints)
    y = foo(x)
    dy_dx = foo_jac(x)[endia.Array]
    d2y_dx2 = foo_hes(x)[endia.Array]

    testing.assert_equal(y.size(), 1)
    testing.assert_equal(y.load(0), 14.0)

    testing.assert_equal(dy_dx.size(), 3)
    testing.assert_equal(dy_dx.load(0), 2.0)

    testing.assert_equal(d2y_dx2.size(), 9)
    testing.assert_equal(d2y_dx2.load(0), 2.0)
    testing.assert_equal(d2y_dx2.load(1), 0.0)


