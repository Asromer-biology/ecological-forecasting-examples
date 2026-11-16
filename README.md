# Ecological forecasting examples

Short, runnable examples of methods I use for ecological forecasting and inference. Each folder goes with one post and stands on its own. Everything runs on simulated or published data, so there is nothing to request before you can run it.

I'm Alex Romer, a quantitative ecologist. These are the small versions of things I've needed in real projects, cut down so the idea is easy to see and quick to run.

## Examples

| Folder | What it shows |
|---|---|
| [`tuned-loss-functions/`](tuned-loss-functions/) | Tuning the settings of a loss function along with the network. A Tweedie power and the two focal loss settings are searched with Keras Tuner on simulated surveys that are mostly zeros. |

## Running an example

Each example is an R script that runs from the repository root, for example

```bash
Rscript tuned-loss-functions/tune_loss_functions.R
```

The README in each folder lists the packages it needs, how long it takes and what it prints.

## License

Code is released under the MIT License (see `LICENSE`).
