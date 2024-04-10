# pgpe_light

A watered down verison of the excellent [nnaisense/pgpelib](https://github.com/nnaisense/pgpelib), that I use in small projects:

```python
from pgpelib.pgpe import PGPE

solver = PGPE(
        solution_length=SIZE,
        popsize=args.pop,
        optimizer="clipup",
        optimizer_config=dict(max_speed=0.25, momentum=0.9),  # .25 .9
        center_learning_rate=0.1,  # 0.15,
        stdev_init=0.05,  # 0.05,
        dtype=np.float32,
        center_init=center,
    )
```
