import blosc2

# Create arrays with specific dimensions
a = blosc2.full((2, 3, 4), 1, urlpath="a.b2nd", mode="w")    # 3D array with dimensions (2, 3, 4)
b = blosc2.full((2, 4), 2,  urlpath="b.b2nd", mode="w")      # 2D array with dimensions (2, 4)
c = blosc2.full(4, 3, urlpath="c.b2nd", mode="w")            # 1D array with dimensions (4,)

print("Array a:", a[:])
print("Array b:", b[:])
print("Array c:", c[:])

# Define an expression using the arrays above
# expression = "a.sum() + b * c"
# expression = "a.sum(axis=1) + b * c"
expression = "sum(a, axis=1) + b * c"
# Define the operands for the expression
operands = {'a': a, 'b': b, 'c': c}
# Create a lazy expression
lazy_expression = blosc2.lazyexpr(expression, operands)

# Store and reload the expressions
url_path = "my_expr.b2nd"  # Define the path for the file
lazy_expression.save(urlpath=url_path, mode="w")  # Save the lazy expression to the specified path

url_path = "my_expr.b2nd"  # Define the path for the file
lazy_expression = blosc2.open(urlpath=url_path)  # Open the saved file
print(lazy_expression)  # Print the lazy expression
print(lazy_expression.shape)  # Print the shape of the lazy expression
print(lazy_expression[:])  # Evaluate and print the result of the lazy expression (should be a 2x4 arr)

# Enlarge the arrays and re-evaluate the expression
a.resize((3, 3, 4))
a[2] = 3
b.resize((3, 4))
b[2] = 5
lazy_expression = blosc2.open(urlpath=url_path)  # Open the saved file
print(lazy_expression.shape)
result = lazy_expression.compute()
print(result[:])
