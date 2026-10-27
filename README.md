# Stochastic Gradient Descent (SGD)

## The Problem with Batch Gradient Descent

Batch Gradient Descent computes the gradient over the **entire dataset** for every single parameter update.

Example: 1000 rows, 5 features → 6 coefficients (5 weights + 1 bias).
- Each update needs gradients computed over all 1000 rows, for all 6 coefficients.
- One epoch = **one** update, and each update costs a full pass over the data.
- On large datasets this is slow, and the full dataset may not fit in memory (hardware limits).

## What is SGD?

Instead of using all rows, SGD updates the coefficients using **one randomly selected row** at a time.

```
for each epoch:
    shuffle the data
    for each row (x_i, y_i):
        gradient = ∇ Loss(x_i, y_i)
        w = w - learning_rate * gradient
```

- **Stochastic** = the row is picked randomly, so each gradient is a noisy estimate of the true gradient.
- 1000 rows → 1000 updates per epoch (instead of 1 in batch GD).
- Each update is cheap, so progress starts immediately → **faster convergence** on large data.
- Trade-off: the path is noisy and fluctuates around the minimum instead of settling smoothly.

## Learning Schedules

Because of this fluctuation, SGD uses a **learning schedule**: the learning rate **decreases as epochs increase**.

- Early epochs: large learning rate → big steps, fast progress.
- Later epochs: small learning rate → small steps, settles near the minimum.

Example: `lr = lr₀ / (1 + decay * epoch)`

## When to Use SGD

- Large datasets (batch GD is too slow or won't fit in memory)
- Non-convex loss functions (the noise helps escape shallow local minima and saddle points)
- When faster initial convergence matters

## Quick Comparison

| | Batch GD | SGD |
|---|---|---|
| Rows per update | All | 1 |
| Updates per epoch | 1 | N (number of rows) |
| Cost per update | High | Low |
| Path to minimum | Smooth | Noisy |
| Large data | Poor | Good |
