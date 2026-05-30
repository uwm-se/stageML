# Real trace schema for StageML MoE LoRA evaluation

Use a JSONL file. Each line is one real request.

```json
{"prompt":"Write a refund email for a damaged package","adapter":"support_adapter"}
```

The adapter value must match the directory name of a real LoRA adapter passed to the benchmark.

The benchmark does not fabricate adapter labels by default. This is intentional so that the evaluation can be based on real tenant or application traces.

Recommended fields

```json
{"timestamp":1710000000.0,"tenant":"tenant_a","adapter":"support_adapter","prompt":"...","expected_output_length":128}
```

Only prompt and adapter are required by the current benchmark.
