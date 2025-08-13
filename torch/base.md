## nn.Linear

- 乘法实际运算:
```python
a = nn.Linear(...)
x @ a.weight.T + a.bias - a(x)
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], ...
```

