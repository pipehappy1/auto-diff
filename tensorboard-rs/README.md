# Write to tensorboard in Rust #

Write TensorBoard events in Rust.

* Can write `scalar`, `image`, `histogram`.

## Example

* Write multiple scalar in one plot. 

```rust,no_run
	let mut writer = SummaryWriter::new(&("./logdir".to_string()));

    for n_iter in 0..100 {
        let mut map = HashMap::new();
        map.insert("x1".to_string(), (n_iter as f32));
        map.insert("x^2".to_string(), (n_iter as f32) * (n_iter as f32));
        writer.add_scalars("data/scalar_group", &map, n_iter);
    }
	writer.flush();
```

