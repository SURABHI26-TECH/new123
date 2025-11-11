
import torch
import numpy as np

# --- Tensor creation and rank/shape ---
t1 = torch.tensor([1, 2, 3, 4])
t2 = torch.tensor([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print("Tensor t1:\n", t1)
print("Tensor t2:\n", t2)

print("\nRank of t1:", len(t1.shape))
print("Shape of t1:", t1.shape)
print("Rank of t2:", len(t2.shape))
print("Shape of t2:", t2.shape)

# --- Creating tensors from data ---
data1 = [1, 2, 3]
data2 = np.array([1.5, 3.4, 6.8, 9.3, 7.0, 2.8])

t1 = torch.tensor(data1)
t2 = torch.Tensor(data1)
t3 = torch.as_tensor(data2)
t4 = torch.from_numpy(data2)

print("\nT1:", t1, "dtype =", t1.dtype)
print("T2:", t2, "dtype =", t2.dtype)
print("T3:", t3, "dtype =", t3.dtype)
print("T4:", t4, "dtype =", t4.dtype)

# --- Reshape, transpose operations ---
t = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

print("\nOriginal Tensor:\n", t)

reshaped = t.reshape(6, 2)
print("\nAfter reshape(6,2):\n", reshaped)

resized = t.reshape(2, 6)
print("\nSimulated resize(2,6) using reshape:\n", resized)

transposed = t.transpose(0, 1)
print("\nAfter transpose(0,1):\n", transposed)

# --- Arithmetic operations ---
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([5, 6, 7, 8])

print("\na:", a)
print("b:", b)

print("\nAddition (a + b):", torch.add(a, b))
print("Subtraction (b - a):", torch.sub(b, a))
print("Multiplication (a * b):", torch.mul(a, b))
print("Division (b / a):", torch.div(b, a))

# --- Autograd example ---
t1 = torch.tensor(1.0, requires_grad=True)
t2 = torch.tensor(2.0, requires_grad=True)

z = 100 * t1 * t2
print("\nExpression z = 100 * t1 * t2 =", z.item())

z.backward()

print("Gradient w.r.t t1:", t1.grad)
print("Gradient w.r.t t2:", t2.grad)
