
# DeepSpeed



## Testing

DeepSpeed tracks two types of tests: unit tests and more costly model convergence tests.
Model convergence tests run
[DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/) and measure
end-to-end convergence and related metrics.

DeepSpeed uses [PyTest](https://docs.pytest.org/en/latest/) to execute tests. PyTest can
be installed from PyPI via `pip install pytest`.  Unit tests are found in `tests/unit/`
and model convergence tests are found in `tests/model/`.

Simply invoke `pytest` to run the unit tests:

    pytest tests/unit/

You can also provide the `-v` flag to `pytest` to see additional information about the
tests.



## Contributing

DeepSpeed welcomes your contributions!

### Prerequisites

DeepSpeed uses [pre-commit](https://pre-commit.com/) to ensure that formatting is
consistent across DeepSpeed.  First, ensure that `pre-commit` is installed from either
installing DeepSpeed or `pip install pre-commit`.  Next, the pre-commit hooks must be
installed once before commits can be made:

    pre-commit install

Afterwards, our suite of formatting tests run automatically before each `git commit`. You
can also run these manually:

    pre-commit run --all-files


### Contributor License Agreement

This project welcomes contributions and suggestions.  Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to, and
actually do, grant us the rights to use your contribution. For details, visit
https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply
follow the instructions provided by the bot. You will only need to do this once across
all repos using our CLA.

### Code of Conduct

This project has adopted the [Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/).  For more information see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or
comments.
