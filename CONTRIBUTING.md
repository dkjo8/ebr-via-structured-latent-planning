## Contributing

Thanks for your interest in improving EBRM.

- **Keep changes small and reviewable**: prefer focused PRs that touch a single concern.
- **Reproducibility**: if you change defaults in `config.toml`, explain the motivation and impact on runs.
- **Generated outputs**: do not commit `runs/`, `modal_output/`, or `analysis/figures/` (they are ignored by git).

### Quick checks

```bash
./scripts/run_tests.sh
```

