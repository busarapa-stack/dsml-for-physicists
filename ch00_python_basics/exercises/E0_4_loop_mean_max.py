"""E0.4  Mean and maximum of the periods with a loop, checked against sum() and max()."""
t10_s = [17.66, 18.02, 17.76, 17.93, 17.93, 18.28, 17.97, 17.97, 17.90, 18.29]
T = [t / 10 for t in t10_s]
total, largest = 0.0, T[0]
for x in T:
    total += x
    if x > largest:
        largest = x
mean = total / len(T)
print(f"loop    : mean = {mean:.4f} s  max = {largest:.3f} s")
print(f"builtin : mean = {sum(T) / len(T):.4f} s  max = {max(T):.3f} s")
