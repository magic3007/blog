---
layout: article
title: "Solutions to Tensor Puzzlers: A Deep Dive into Tensor Operations"
date: 2025-02-10 12:00:00 +0800
categories: [Programming, Machine Learning]
tags: [pytorch, numpy, tensor, programming-challenges]
---

# Solutions to Tensor Puzzlers: A Deep Dive into Tensor Operations

This article presents my solutions to [Sasha Rush's Tensor Puzzles](https://github.com/srush/Tensor-Puzzles), a collection of challenging tensor manipulation problems. These solutions serve as an excellent reference for understanding fundamental tensor operations and their creative applications.

## Foundation: Essential Operations

Before diving into the solutions, let's understand two fundamental operations that form the backbone of many tensor manipulations:

```python
def arange(i: int):
    """Generate a sequence of numbers from 0 to i-1.
    Used as a replacement for traditional for-loops."""
    return torch.tensor(range(i))

def where(q, a, b):
    """Conditional element selection.
    Acts as a vectorized if-statement: returns a when q is True, b otherwise."""
    return (q * a) + (~q) * b
```

These operations enable two primary paradigms in tensor manipulation:
1. **Masking**: Using `where` to selectively modify tensor elements
2. **Matrix Operations**: Combining `arange` and `where` for dimension manipulation and broadcasting

## Basic Tensor Operations

### 1. Creating a Tensor of Ones
```python
def ones(i: int) -> TT["i"]:
    """Generate a 1D tensor filled with ones."""
    return where(arange(i) >= 0, 1, 0)
```
This elegant solution leverages broadcasting to create a tensor of ones by comparing `arange` output with 0.

### 2. Computing Sum Along Axis
```python
def sum(a: TT["i"]) -> TT[1]:
    """Compute the sum of all elements in a tensor."""
    return ones(a.shape[0]) @ a[:, None]
```
This implementation uses matrix multiplication with a vector of ones, demonstrating an alternative to traditional reduction operations.

### 3. Outer Product
```python
def outer(a: TT["i"], b: TT["j"]) -> TT["i", "j"]:
    """Compute the outer product of two vectors."""
    return a[:, None] @ b[None, :]
```
The solution uses broadcasting through dimension expansion to create a 2D tensor representing the outer product.

## Advanced Matrix Operations

### 4. Diagonal Matrix Operations
```python
def diag(a: TT["i", "i"]) -> TT["i"]:
    """Extract the diagonal elements of a matrix."""
    return a[arange(a.shape[0]), arange(a.shape[0])]
```

### 5. Identity Matrix
```python
def eye(j: int) -> TT["j", "j"]:
    """Create an identity matrix of size j×j."""
    return where(arange(j)[:, None] == arange(j)[None, :], 1, 0)
```

### 6. Upper Triangular Matrix
```python
def triu(j: int) -> TT["j", "j"]:
    """Create an upper triangular matrix."""
    return where(arange(j)[:, None] <= arange(j)[None, :], 1, 0)
```

## Sequence Operations

### 7. Cumulative Sum
```python
def cumsum(a: TT["i"]) -> TT["i"]:
    """Compute cumulative sum of elements."""
    return (a[:, None] * (arange(a.shape[0])[:, None] <= arange(a.shape[0]))).sum(0)
```

### 8. Element-wise Difference
```python
def diff(a: TT["i"], i: int) -> TT["i"]:
    """Compute differences between adjacent elements."""
    return where(arange(i) == 0, a, a - a[arange(i)-1])
```

## Array Manipulation

### 9. Vertical Stack
```python
def vstack(a: TT["i"], b: TT["i"]) -> TT[2, "i"]:
    """Stack two arrays vertically."""
    return where(arange(2)[:, None] != ones(a.shape[0]), a, b)
```

### 10. Array Rolling
```python
def roll(a: TT["i"], i: int) -> TT["i"]:
    """Shift array elements cyclically."""
    return a[(arange(i)+1)%i]
```

### 11. Array Reversal
```python
def flip(a: TT["i"], i: int) -> TT["i"]:
    """Reverse the order of array elements."""
    return a[i-1-arange(i)]
```

## Advanced Operations

### 12. Array Filtering
```python
def compress(g: TT["i", bool], v: TT["i"], i:int) -> TT["i"]:
    """Filter array elements based on boolean mask."""
    return v @ where(g[:,None], arange(i) == cumsum(1*g)[:,None]-1, 0)
```

### 13. Array Padding
```python
def pad_to(a: TT["i"], i: int, j: int) -> TT["j"]:
    """Pad array to specified length."""
    return a @ where(arange(i)[:, None] == arange(j), 1, 0)
```

### 14. Sequence Masking
```python
def sequence_mask(values: TT["i", "j"], length: TT["i", int]) -> TT["i", "j"]:
    """Create attention mask for sequence operations."""
    return where(length[:,None] > arange(values.shape[1]), values, 0)
```

### 15. Element Counting
```python
def bincount(a: TT["i"], j: int) -> TT["j"]:
    """Count occurrences of each value in the input array."""
    # Two equivalent implementations:
    # return ones(a.shape[0]) @ where(arange(j) == a[:, None], 1, 0)
    # Using identity matrix for indexing
    return ones(a.shape[0]) @ eye(j)[a]
```

### 16. Scatter Addition
```python
def scatter_add(values: TT["i"], link: TT["i"], j: int) -> TT["j"]:
    """Add values to specified positions in output array."""
    # Two equivalent implementations:
    # return values @ where(link[:, None] == arange(j), 1, 0)
    # Using identity matrix for index operations
    return values @ eye(j)[link]
```

### 17. Array Flattening
```python
def flatten(a: TT["i", "j"], i:int, j:int) -> TT["i * j"]:
    """Flatten a 2D array into 1D."""
    return a[arange(i*j)//j, arange(i*j)%j]
```

### 18. Linear Space Generation
```python
def linspace(i: TT[1], j: TT[1], n: int) -> TT["n", float]:
    """Generate evenly spaced numbers over a specified interval."""
    return (i + (j - i) * arange(n) / max(1, n - 1))
```

### 19. Heaviside Step Function
```python
def heaviside(a: TT["i"], b: TT["i"]) -> TT["i"]:
    """Implement the Heaviside step function."""
    return where(a==0, b, where(a>0,1,0))
```

### 20. Array Repetition
```python
def repeat(a: TT["i"], d: TT[1]) -> TT["d", "i"]:
    """Repeat array elements along a new axis."""
    return ones(d)[:, None] @ a[None, :]
```

### 21. Value Bucketing
```python
def bucketize(v: TT["i"], boundaries: TT["j"]) -> TT["i"]:
    """Assign values to buckets based on boundaries."""
    return ones(boundaries.shape[0]) @ where(boundaries[:, None] <= v, 1, 0)
```

## Key Insights and Patterns

1. **Index Operations**: Many operations can be transformed into matrix operations using the identity matrix (`eye`).
2. **Broadcasting**: Clever use of broadcasting can simplify complex operations.
3. **Masking**: The `where` operation provides a powerful way to implement conditional logic.

## Implementation Tips

1. Use `ones(j)[a]` as an alternative to `arange(j) == a[:, None]`
2. Leverage broadcasting to avoid explicit loops
3. Consider matrix multiplication for sequence operations
4. Transform index operations into matrix operations using the identity matrix

## References

1. [Tensor Puzzlers Repository](https://github.com/srush/Tensor-Puzzles)
2. [PyTorch Broadcasting Documentation](https://pytorch.org/docs/stable/notes/broadcasting.html)
3. [NumPy Broadcasting Documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html)

For complete implementation details, check out my [solution gist](https://gist.github.com/magic3007/0a6d1941de58bf73fc0d23ea612e05e3).