# Contribute to BayesBlend

## Dependency Management

BayesBlend recommends using Python `3.11` for local development and contributing.

This project uses [Poetry](https://python-poetry.org/) to manage dependencies. Poetry needs to be installed first, which requires the command line utility `pipx`. 
To ensure the same Python versions are used to install `pipx` and Poetry, we recommend installing Poetry using:

```
python3.11 -m pip install pipx
python3.11 -m pipx install poetry
```

If the default Python version on your machine is already `3.11`, `python3.11 -m` can be ommitted from the
above commands.

Note that Poetry is installed in a dedicated virtual environment by `pipx` separate from the development environment used for this project. Refer to the [Poetry installation docs](https://python-poetry.org/docs/#installing-with-pipx) for further information. 

After installation, use Poetry to install the project and dependencies: 

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

To run Ruff checks: 

```
poetry run ruff check
```

And run MyPy checks: 

```
poetry run mypy .
```

Note that both Ruff and MyPy are required CI/CD checks that will block merging a pull request into main, so it is important to check that they both pass locally before submitting a pull request. 

## Versioning

This project uses semantic versioning, primarily following the [Angular convention](https://gist.github.com/brianclements/841ea7bffdb01346392c). Package releases are auto-generated based on commit messages, so it is important to strictly follow the semantic versioning convention to ensure that releases are deployed as intended. Angular commit messages follow the below template: 

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

Available values for `<type>` include:

- `build`: Changes that affect the build system or external dependencies 
- `ci`: Changes to the CI configuration files and scripts
- `docs`: Documentation only changes
- `feat`: A new feature
- `fix`: A bug fix
- `perf`: A code change that improves performance
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing tests

`<scope>` is optional, and when specified it should refer to the name of the module (or package) being modified per the commit. 

`<description>` should provide a succinct summary of the change (with no period at the end). 

`<body>` is optional in our case, but is strongly recommended. It should explain why the change is being made, including comparisons of previous vs new behavior to clarify the change. 

`<footer>` is optional, and when specified it should contain information about breaking changes, deprecations, and is also a place to reference issues that the PR resolves. For example, `Fixes: #11` if the issues resolves issue 11. Breaking changes and deprecations notices should also be included in the footer along with descriptions to clarify the change/instructions for future use. For example: 

```
BREAKING CHANGE: <breaking change summary>
<BLANK LINE>
<breaking change description + migration instructions>
<BLANK LINE>
<BLANK LINE>
Fixes #<issue number>
```

or 

```
DEPRECATED: <what is deprecated>
<BLANK LINE>
<deprecation description + recommended update path>
<BLANK LINE>
<BLANK LINE>
Closes #<pr number>
```

These conventions should be thought of as guidelines, but we do recommend sticking to them as closely as possible to make the contribution process smooth for all involved. 
