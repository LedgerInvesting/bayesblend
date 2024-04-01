# Contribute to bayesblend

## Dependency Management

This project uses [Poetry](https://python-poetry.org/) to manage dependencies. Poetry needs to be installed first:

```
pipx install poetry
```

Note that `Poetry` should be installed in a dedicated virtual environment separate from the development environment used for this project. Refer to the [Poetry installation docs](https://python-poetry.org/docs/#installing-with-pipx) for further information. 

After installation, use `Poetry` to install the project and dependencies: 

```
poetry install
```

NOTE: Throughout this doc, you will see many commands of the form `poetry ...`. The `poetry` command is just ensuring that `Python` uses the `Poetry`-configured project when runnings commands. Alternatively, you can enter the `Poetry` virtual environment per `poetry shell` and interact with `Python` as per usual without pre-fixing with `poetry` each time. For example, in the `Poetry` shell, `poetry run python ...` simply becomes `python ...`. 

## Run Tests

To ensure that the project is installed and functioning properly, run the tests and check that they pass:

```
poetry run pytest
```

## Code Style and Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) as a linter and formatter and [MyPy](https://mypy-lang.org/) for static type checking. Both are listed in the `dev` group of dependencies that should get installed per `poetry install` (refer to the `pyproject.toml` to see versions). 

To run `Ruff` checks: 

```
poetry run ruff check
```

And run `MyPy` checks: 

```
poetry run mypy
```

Note that both `Ruff` and `MyPy` are required CI/CD checks that will block merging a pull request into main, so it is important to check that they both pass locally before submitting a pull request. 

## Versioning

TODO