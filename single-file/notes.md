# TODO:

How to verify that Polars chose it

After building your lazy query, inspect the physical plan before collecting:

```
query = left.join(right, on="key", how="inner")
query.show_graph(plan_stage="physical")
```
